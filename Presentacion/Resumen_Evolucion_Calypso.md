# Evolución del Sistema Recomendador Calypso: De la Etapa 1 a Calypso 6.2b

**Equipo:** DataVengers (Team 32)  
**Proyecto:** Datathon FME 2026 · RawTree Real-Time Recommendation Challenge  
**Fecha:** 15 de noviembre de 2026  

---

## 1. Introducción y Contexto del Reto

El reto de RawTree consiste en servir recomendaciones en un comercio electrónico con más de 5.000 productos. Los compradores llegan en tiempo real (~3 peticiones por segundo). Para cada uno, nuestro sistema debe devolver una lista ordenada de **exactamente 10 productos activos y en stock** en menos de **3.000 milisegundos**. Un simulador oculto evalúa las decisiones del comprador en base a sus probabilidades reales de compra.

La evaluación final consta de:
* **70 puntos automáticos:** 50 puntos de Conversión global, 10 puntos de Cold Start (usuarios o productos nuevos) y 10 puntos de Fiabilidad / Reliability (respetar estrictamente el SLA de 3 segundos), con desempate por ingresos (€/petición).
* **30 puntos del Jurado:** Método y validación offline (10p), Aprendizaje del flujo en tiempo real (10p), Uso avanzado de RawTree/SQL (5p) y Claridad de comunicación (5p).

A lo largo de 24 horas ininterrumpidas de desarrollo, nuestro recomendador **Calypso** evolucionó a través de múltiples versiones. A continuación se detalla la historia de cada versión, sus innovaciones, los problemas reales que presentó y las lecciones aprendidas.

---

## 2. Línea Temporal Detallada de Versiones

```
Etapa 1 (Híbrido) ──▶ Etapa 2 (Lift + Perfil) ──▶ Etapa 3 (ML + IPS + Caída SLA)
                                                               │
                                                               ▼
Calypso 3.2 ◀── Calypso 3.1b (Ablación) ◀── Calypso 3.1 ◀── Etapa 4 (Blindaje Caché)
     │
     ▼
Calypso 4.1 / 4.1_e (Reglas Negocio + Error Precio) ──▶ Calypso 4.2 (Corrección M3)
                                                               │
                                                               ▼
Calypso 6.2b (Dual-Head + Student MLP + Cold Focus) ◀── Calypso 5.1 (Modular Hot-Reload)
```

---

### 2.1 Etapa 1: `etapa1-hybrid-v1` (El Origen Desacoplado)

* **Qué introdujo:**
  * **Arquitectura de dos bucles desacoplados:** Un bucle rápido (*Fast Serving Loop*) en memoria pura que responde peticiones en milisegundos y un bucle lento (*Slow Background Loop*) en segundo plano que consulta RawTree sin bloquear la atención al cliente.
  * **Snapshot inmutable en memoria:** Los datos se empaquetan en un objeto estático inmutable; el proceso lector nunca espera bloqueos de base de datos.
  * **Señales iniciales:** Co-visitas en la misma sesión, auto-conversión de productos recién vistos, popularidad contextual por dispositivo y región, filtro TF-IDF de descripciones para productos nuevos y exclusión básica de productos ya comprados en el histórico.
  * **Barrera de guardrails determinista:** Garantía de 10 IDs únicos, activos y en stock antes de cada envío.

* **Problemas y Limitaciones:**
  * **Cuello de botella en el refresco completo:** La función `refresh_full()` recalculaba los 60 días históricos cada 180 segundos. Mientras duraba esa pesada consulta SQL, el refresco de stock se quedaba paralizado; si un producto se agotaba en esos segundos, corría el riesgo de ser recomendado e invalidar la respuesta entera.
  * **Riesgo de violación del SLA:** En el peor caso, la lectura (1.2 s) + el push (1.4 s) + el reintento (1.4 s) sumaban más de 4 segundos, superando el límite de 3 segundos.
  * **Pérdida de peticiones:** La marca `_seen` se aplicaba *antes* de confirmar el push HTTP; si la red fallaba, las peticiones se descartaban y no se reintentaban.
  * **Transiciones sin dirección temporal:** El cruce SQL no exigía que el producto B ocurriera *después* del producto A en el tiempo, y los superventas dominaban todas las asociaciones.
  * **Validación circular:** El evaluador offline usaba un simulador sintético creado con las mismas asunciones del recomendador, por lo que no predecía el comportamiento del simulador real de los organizadores.

---

### 2.2 Etapa 2: `etapa2-seq-mmr-v1` (Lift, Perfiles de Usuario y Cold Start Bidireccional)

* **Qué introdujo:**
  * **Transiciones con Lift debiassado:** Se dividió la frecuencia de co-ocurrencia por la popularidad del producto destino ($\text{score} / \text{pop}^\beta$), evitando que los superventas acapararan todas las sugerencias.
  * **Decaimiento dual en sesión:** Ponderación que combina el orden inverso del producto visto con el tiempo real transcurrido en segundos ($\Delta t$).
  * **Perfil de usuario (`UserProfile`):** Detección de intenciones no compradas (artículos añadidos al carrito o vistos repetidamente), afinidad de marca y estimación gaussiana del rango de precio habitual del usuario.
  * **Cold Start bidireccional (*Graph Grafting*):** No solo de producto nuevo a conocido (si miras un producto nuevo), sino de conocido a nuevo (si miras un artículo conocido, te sugiere productos nuevos semánticamente afines usando TF-IDF + TruncatedSVD).
  * **Diversidad MMR y desempate por ingresos:** Algoritmo Maximum Marginal Relevance en slots 4–10 para evitar listas monótonas y un multiplicador suave para desempatar a favor del producto de mayor precio.

* **Problemas y Limitaciones:**
  * **Explosión de hiperparámetros manuales:** Se pasó de 12 a 22 parámetros puestos a mano sin contrastar con datos reales.
  * **Bloqueo del GIL de Python por sobrecarga de datos:** Las consultas de perfiles de usuario descargaban cientos de miles de filas JSON cada 3 minutos. La decodificación en C bloqueaba el GIL (*Global Interpreter Lock*), congelando brevemente el hilo que respondía las peticiones en vivo.
  * **Normalización engañosa en Cold Start:** Al normalizar la similitud por el máximo de cada producto, el primer producto nuevo obtenía siempre similitud 1.0, vinculándose erróneamente a productos de subcategorías completamente distintas.
  * **Falso desempate por ingresos:** El multiplicador de precio reordenaba cualquier par con menos de un 6% de diferencia a favor del más caro, empeorando la conversión si el comprador buscaba precios moderados.
  * **Excesivo foco en usuarios con historial:** Casi toda la ingeniería se dedicó a usuarios conocidos, cuando la final se centraría en compradores y catálogo nuevos.

---

### 2.3 Etapa 3: `calypso-2.1` (Machine Learning con IPS, Bandits y Agente Autónomo)

* **Qué introdujo:**
  * **Re-Ranker Logístico con IPS:** Primer modelo de Machine Learning supervisado. Evaluaba 14 variables normalizadas por candidato y ponderaba las muestras históricas por el inverso de la propensión posicional $((k+1)^{0.75})$ para corregir el sesgo del carrusel antiguo.
  * **Bandit UCB posicional:** Estimación de la tasa de conversión empírica por producto descontando la posición del slot en que se mostró, añadiendo un bonus de exploración Upper Confidence Bound (UCB).
  * **Agente Autónomo en `scratch`:** Un agente en segundo plano (`CalypsoAgent`) analizaba el CTR/CVR y las oleadas de productos nuevos en vivo, adaptando hiperparámetros y registrando sus decisiones en `team_32.scratch`.

* **El Gran Problema de la Etapa 3: La Caída del SLA al 10.3%:**
  * Tras un reinicio del ordenador, el sistema experimentó una caída catastrófica de disponibilidad. El panel mostró un **10.3% de ON TIME**.
  * **Causa 1 (Peticiones zombi en cola):** La consulta `SQL_PULL` traía `LIMIT 500 ORDER BY ts DESC`. Unas 490 de esas 500 peticiones ya habían caducado hacía minutos (>3.000 ms). El sistema gastaba más de un segundo en procesar y enviar respuestas muertas, acumulando un retraso permanente sobre las peticiones frescas que iban entrando.
  * **Causa 2 (Arranque frío bloqueante):** Al arrancar sin memoria persistente local, debía ejecutar 11 consultas masivas sobre 12 millones de registros de RawTree, tardando decenas de segundos donde todas las peticiones caían en fallback.
  * **Causa 3 (Invasión de slots WARM):** El modo agresivo de cold-start colocaba productos nuevos no validados en los slots 0–3, degradando la conversión de los usuarios recurrentes.

---

### 2.4 Etapa 4: El Blindaje Operativo Anti-Caídas (La Base Moderna)

* **Qué introdujo:**
  * **Caché atómica a disco:** Guardado periódico del estado completo en disco (`snapshot_v4_latest.pkl`). Cualquier reinicio o fallo del proceso permitía volver a servir recomendaciones en **menos de 0.4 segundos**.
  * **Lectura exclusiva de peticiones frescas:** Consulta SQL estrictamente acotada a `ts >= now64(3) - 2400ms ORDER BY ts ASC LIMIT 25`. Cero milisegundos malgastados en peticiones muertas.
  * **Confirmación `_seen` post-push:** La deduplicación solo marca la petición como vista *después* de que RawTree confirme la recepción correcta del lote.
  * **Supervisor Watchdog:** Script en PowerShell (`run_watchdog.ps1`) y bucle en Python que relanzan el recomendador en 1 segundo si el proceso se detiene.
  * **Separación de histórico estático y flujo en vivo:** Las consultas de 60 días se realizan una única vez y se cachean. En vivo solo se ejecutan dos tipos de consultas ligeras: stock cada 1.5 s y eventos/feedback cada 15 s.
  * **Arquitectura Slot-Gated:** Slots 0–6 blindados exclusivamente para máxima precisión WARM y coherencia de sesión; slots 7–9 reservados para exploración UCB y productos nuevos (máximo 2 nuevos).

* **Resultado:** La fiabilidad subió de inmediato por encima del **99.5% ON TIME**, devolviendo el sistema al rango competitivo del leaderboard.

---

### 2.5 Calypso 3.1: El Tropiezo de la Agresividad

* **Qué intentó hacer:**
  * Quiso aplicar cuatro cambios drásticos a la vez sobre 2.1: eliminar los suelos mínimos de afinidad de subcategoría/marca, aplicar una fuerte penalización de fondo a productos sin señal de sesión (x0.45 en misma subcategoría, x0.18 en distinta), introducir un modelo colaborativo EASE con peso 0.25 y forzar consultas SQL direccionales.

* **Problemas y Regresión:**
  * **Caída del rendimiento:** El porcentaje de acierto a compra cayó del **77.2% al 74.5% (-2.7%)**.
  * **Lección crítica:** Cambiar múltiples piezas simultáneamente impide aislar los fallos. Al quitar los suelos de subcategoría, el sistema forzaba productos irrelevantes (-2.0%). El modelo EASE con peso alto introducía ruido matemático (-0.8%). El único cambio positivo fue la penalización de fondo (+1.8%).

---

### 2.6 Calypso 3.1b: La Modularización Inteligente y la Ablación

* **Qué introdujo:**
  * Convirtió cada cambio de 3.1 en un interruptor configurable por variable de entorno (`CALYPSO_V3_*`).
  * Ejecutó un estudio sistemático de ablación con 5 semillas aleatorias para medir el impacto individual de cada hipótesis.
  * Dejó activada por defecto únicamente la **penalización de fondo** (`v3_bg_penalty = True`) y apagó el resto.

* **Resultado:**
  * Marcó el récord del proyecto en ese momento con un **79.0% de acierto a compra** (+1.8% sobre Calypso 2.1), confirmando la regla de oro: *"No recomiendes lo que no tenga un motivo contrastado para ser recomendado"*.

---

### 2.7 Calypso 3.2: Semividas Temporales y Segmentación

* **Qué introdujo:**
  * **Semivida temporal estricta (624 segundos):** El peso de cada producto visto decae a la mitad cada 10 minutos, priorizando la intención inmediata del comprador.
  * **Prior bayesiano jerárquico ($\kappa = 50$):** Suavizado de popularidad contextual entre contexto específico, regional y global.
  * **Segmentación en 5 perfiles:** Reglas diferenciadas para comprador nuevo con sesión, nuevo sin sesión, conocido con sesión, conocido sin sesión y producto nuevo.

* **Transición:** Demostró mejoras cualitativas, pero evidenció la necesidad de formalizar reglas de negocio comerciales (carritos y recompras) y de un sistema de gobierno centralizado de pesos.

---

### 2.8 Calypso 4.1 y 4.1_e: Reglas de Negocio y el Descubrimiento del Orden por Precio

* **Qué introdujo:**
  * Cinco reglas de negocio explícitas conmutables por entorno (`CALYPSO_V4_*`):
    * **M1 (Carrito activo):** Productos añadidos al carrito hace $\le 30$ min garantizados en los primeros slots (máx. 3).
    * **M2 (Carrito olvidado):** Productos en carrito de hace más de 30 min colocados justo tras el carrito activo (máx. 2).
    * **M3 (Orden por precio más barato):** Ordenar cada bloque de productos por precio ascendente (`v4_price_ascending = True`).
    * **M4 (Recurrentes garantizados):** Productos comprados $\ge 2$ veces por el usuario no se excluyen y se garantizan en los recomendados.
    * **M5 (Compras orgánicas excluidas):** Las compras de `shop.events` también se consideran para no repetir recomendaciones.
  * **Versión 4.1_e:** Primera versión con `weights_profile.json` centralizado y herramienta `tuner.py`.

* **El Gran Problema de 4.1: El Desastre del Orden por Precio:**
  * **Pérdida del 25% de DCG:** El estudio de ablación destapó que ordenar los productos por precio ascendente (`v4_price_ascending = True`) **reducía el DCG de compra un 25%** (de 0.485 a 0.365).
  * **Causa:** El Slot 1 concentra el **34%** de las compras y los slots 1–3 el **63%**. Al forzar los productos más baratos arriba, se apartaba del primer puesto al producto con mayor probabilidad real de compra estimada por el modelo, mandándolo a posiciones inferiores donde casi nadie compra.
  * **Lección:** El precio solo puede usarse como un **desempate suave** entre productos con puntuaciones prácticamente idénticas dentro de la misma subcategoría, jamás como un criterio de ordenación global. Además, reservar 3 slots fijos al carrito desplazaba predicciones superiores de Machine Learning.

---

### 2.9 Calypso 4.2: La Corrección de las Medidas de Conversión

* **Qué introdujo:**
  * Implementó las correcciones del informe `MEDIDAS_CONVERSION.md`:
    * Apagó inmediatamente el orden por precio (`v4_price_ascending = False`).
    * Sustituyó el bloqueo rígido de slots de carritos por bonificaciones de puntuación para que compitieran de forma natural.
    * Introdujo la **tasa base explícita por ranking de ventas** ($1 - \ln(\text{rank}) / \ln(\text{max\_rank})$ combinada con compras ∝ $\text{puesto}^{-0.87}$).
    * Redujo la diversidad forzada en la página principal (`mmr_subcat_lambda_home = 0.0`), ya que dispersar categorías en una tienda sin datos perjudicaba la conversión.

---

### 2.10 Calypso 5.1: Gobernanza Modular Completa y Recarga en Caliente

* **Qué introdujo:**
  * **Perfil unificado con Hot-Reload (`weights_profile.json`):** Unificación de los 14 pesos prior, coeficientes por segmento de usuario, multiplicadores de página y penalizaciones. El servicio detecta cambios en el archivo y recarga los pesos en caliente sin detener el bucle rápido ni perder peticiones.
  * **Pool de candidatos ampliado (Top-35 ventas garantizado):** Genera hasta 140 candidatos asegurando que los 35 superventas globales y contextuales siempre estén presentes antes de calcular transiciones.
  * **Co-visitas moduladas por tasa base:** Evita que co-visitas marginales de productos raros superen a productos que realmente venden.
  * **Eliminación de constantes hardcoded:** Se suprimió la penalización fija de x0.25 y se parametrizó la compatibilidad de precio.

* **Limitaciones:**
  * El ajuste de los ~65 parámetros dependía de búsquedas heurísticas sobre las 386 compras reales del log descargado, alcanzando un techo de optimización. No contaba con un modelo no lineal que corrigiera los sesgos residuales sobre el histórico completo.

---

### 2.11 Calypso 6.2b: La Versión Definitiva para la Ronda Final

* **Qué introduce (La cúspide técnica del equipo):**
  1. **Arquitectura Dual-Head Slot-1 Precision:** Cabecera de red dedicada específicamente al Slot 1. Utiliza 64 variables y regularización del riesgo secuencial sobreestimado para asegurar la máxima precisión en el slot que concentra el 34% de las conversiones de la tienda.
  2. **Transfer Learning en 2 Fases (Teacher → Student MLP):** Entrena un clasificador Teacher avanzado (`HistGradientBoostingClassifier`) sobre 13.129 compras históricas sin fuga de datos y lo destila en un Student MLP en NumPy puro con capas `(80, 40)` y ajuste fino con regularización Ridge.
  3. **Multi-Source IPS Learning:** Aprendizaje multi-fuente que combina compras Gold con ponderación IPS (`1.0`), carritos activos (`0.18`), intenciones recurrentes (`0.14`) y transiciones de sesión (`0.12`), empleando *hard negatives* extraídos del entorno real de inferencia.
  4. **Transiciones cruzadas entre subcategorías (`cross_subcat_lift`):** Modela la matriz $P(\text{subcat}_B \mid \text{subcat}_A)$ para capturar ventas cruzadas de productos complementarios entre categorías distintas.
  5. **Live-CVR Velocity:** Monitorización en tiempo real de la velocidad de conversión desde RawTree para detectar tendencias repentinas del domingo.
  6. **Vectorización FastLoop (< 2.8 ms):** Cómputo matricial directo en NumPy que garantiza respuestas en menos de 3 milisegundos de forma constante.
  7. **Rendimiento contrastado:** Assoleix un **74.41% de compra global**, lidera el Slot 1 con **0.2709 de SNIPS CVR** y alcanza un extraordinario **82.61% en Cold Start** (con el preset `product_profile_focus`).

---

## 3. Cuadro Resumen Comparativo de Todas las Versiones

| Versión | Innovación Principal | Problema Crítico Presentado | Resultado / Lección |
| :--- | :--- | :--- | :--- |
| **Etapa 1** | Dos bucles desacoplados + Guardrails | Refresco de 180s bloqueaba stock; peor caso >4s | La arquitectura base funcionó, pero requería optimización de I/O y tiempos. |
| **Etapa 2** | Lift debiassado + Perfil de usuario | Parseo JSON masivo bloqueaba el GIL; 22 pesos a ciegas | La personalización no debe sobrecargar la memoria ni ignorar a usuarios nuevos. |
| **Calypso 2.1** | Re-Ranker ML con IPS + Bandits | **Caída de SLA al 10.3%** por procesar peticiones zombi y reiniciar sin caché | Imprescindible guardar estado en disco y filtrar por edad de petición (<2.4s). |
| **Etapa 4** | Caché atómica (<0.4s) + Slot-Gating | Ninguno crítico en operación; pesos aún heurísticos | Se blindó la fiabilidad al 99.5%+ ON TIME y la recuperación instantánea. |
| **Calypso 3.1** | Quitar suelos + EASE colaborativo | **Regresión del 77.2% al 74.5%** por cambios simultáneos bruscos | EASE con peso alto añade ruido; forzar categorías sin datos resta ventas. |
| **Calypso 3.1b** | Interruptores `CALYPSO_V3_*` + Ablación | Faltaban reglas de negocio para carritos y recompras | **Récord intermedio de 79.0%** aislando solo la penalización de fondo. |
| **Calypso 3.2** | Semivida de 10 min + Bayes contextual | Transición intermedia sin motor de gobierno centralizado | Necesidad de desacoplar configuración de código mediante perfiles JSON. |
| **Calypso 4.1** | Reglas de negocio (Carrito, Recurrentes) | **Pérdida del 25% de DCG** por ordenar de más barato a más caro | El precio jamás debe mandar sobre la relevancia en el Slot 1. |
| **Calypso 4.2** | Fin de orden por precio + Tasa base | Búsqueda limitada a pocas compras del log descargado | La tasa base de ventas es el ancla más sólida de conversión. |
| **Calypso 5.1** | Gobernanza modular con Hot-Reload | Techo de optimización manual de 65 parámetros | La recarga en caliente permite adaptar el modelo sin parar el servicio. |
| **Calypso 6.2b** | Dual-Head Slot-1 + Destilación MLP | Complejidad de 190 parámetros (resuelta con presets) | **Modelo definitivo:** 74.4% de compra general y 82.6% en Cold Start. |

---

## 4. Argumentos Clave para la Presentación ante el Jurado (30 Puntos)

1. **Método y Validación Offline (10 puntos):**
   * Demostramos rigor científico mediante estudios sistemáticos de ablación (ej. el descubrimiento del coste del orden por precio en 4.1 y el aislamiento de la penalización de fondo en 3.1b).
   * No nos fiamos de simuladores sintéticos circulares: validamos con particiones temporales estrictas sin fuga de datos (*train* anterior a corte, *test* posterior).
2. **Aprendizaje del Flujo en Tiempo Real (10 puntos):**
   * El sistema no es estático: aprende continuamente mediante el bandit posicional con corrección de atención por slot $((k+1)^{-0.75})$, el detector de velocidad de CVR en vivo (`shop.events`) y la exclusión inmediata de compras orgánicas.
   * Manejo superior de Cold Start: inyección de productos nuevos en slots controlados (5–9) y herencia semántica TF-IDF + SVD desde el primer segundo.
3. **Uso Avanzado de RawTree / SQL (5 puntos):**
   * Cómputo inteligente de agregaciones complejas directamente en ClickHouse (co-visitas con Lift temporal, ventanas móviles de 60 min, deduplicación de carritos).
   * Empleo activo de `team_32.scratch` para persistir decisiones del agente, pesos del modelo y trazabilidad de experimentos.
4. **Resiliencia Operativa e Ingeniería (Fiabilidad):**
   * Tras la incidencia de la Etapa 3, blindamos la infraestructura: caché atómica a disco, arranque en <0.4s, descarte de peticiones caducadas y watchdog permanente con **>99.5% de cumplimiento de SLA**.
