# Plan de acción: User-Based, Item-Based y modelo híbrido de Collaborative Filtering

**Contexto:** RawTree Challenge · Datathon FME 2026\
**Objetivo de este documento:** definir cómo explorar los datos y
desarrollar tres experimentos comparables de recomendación:

1.  **User-Based Collaborative Filtering (User-CF)**, desarrollado de
    forma independiente.
2.  **Item-Based Collaborative Filtering (Item-CF)**, desarrollado de
    forma independiente.
3.  **Modelo combinado User-CF + Item-CF**, que aprovecha las
    predicciones de ambos.

La idea es construir primero una base de datos de interacciones fiable y
después desarrollar cada variante por separado, medirlas con el mismo
protocolo y, finalmente, probar si la combinación mejora los resultados.

------------------------------------------------------------------------

## 0. Qué sabemos del challenge y qué no debemos asumir

Según el README del challenge, el catálogo tiene alrededor de 5.000
productos y 10 categorías. Hay dos fuentes históricas especialmente
útiles:

### `shop.history_events`

Contiene aproximadamente 12 millones de eventos orgánicos de los últimos
60 días. Los tipos documentados son:

-   `view`
-   `add_to_cart`
-   `purchase`

Los campos documentados incluyen `ts`, `event_type`, `user_id`,
`session_id`, `product_id`, `price_eur`, `device` y `region`.

Estos eventos no proceden del carrusel de recomendaciones. Por eso son
útiles para aprender relaciones entre usuarios y productos sin depender
exclusivamente de lo que el sistema anterior decidió mostrar.

### `shop.history_sessions`

Contiene aproximadamente 2 millones de registros de impresiones del
carrusel de los últimos 60 días. Incluye:

-   `request_id`, `ts`, `user_id`
-   `context` (device, region y page)
-   `recent_views`
-   `shown`: los diez productos que enseñó el sistema anterior
-   `propensity`: probabilidad asociada a cada posición/producto
    mostrado
-   `clicked`, `purchased` y `revenue_eur`

El README advierte que este historial está sesgado hacia productos
populares porque el sistema anterior recomendaba principalmente best
sellers. `propensity` puede ayudar a corregir parte de ese sesgo.

### Datos que llegan durante el evento

-   `shop.requests`: solicitudes nuevas de recomendación,
    aproximadamente 3 por segundo.
-   `shop.events`: eventos orgánicos en tiempo real.
-   `shop.catalog_updates`: altas de productos, cambios de precio,
    productos sin stock y productos que vuelven a estar disponibles.
-   `team_NN.feedback`: resultado de las recomendaciones del equipo.

**Importante:** el README describe las tablas y sus campos, pero no
garantiza que todas las consultas, agregaciones o transformaciones estén
ya implementadas. Primero hay que inspeccionar los datos reales y
validar los tipos, valores nulos, duplicados y formatos.

------------------------------------------------------------------------

# 1. Fase común: exploración y preparación de datos

Esta fase se realiza una sola vez y sirve para los tres experimentos. No
conviene empezar calculando similitudes antes de conocer la calidad y
estructura de los datos.

## 1.1. Comprobar el acceso y el contenido de las tablas

Objetivo: verificar que el equipo puede consultar las tablas compartidas
y que los datos tienen el formato esperado.

Acciones:

1.  Configurar las credenciales de RawTree siguiendo las instrucciones
    del starter kit.
2.  Consultar el número de registros de `shop.catalog`,
    `shop.history_events` y `shop.history_sessions`.
3.  Inspeccionar una muestra de registros de cada tabla.
4.  Comprobar qué tipos de datos tienen los identificadores y las marcas
    temporales.
5.  Confirmar cómo se representan los valores ausentes: `null`, cadena
    vacía u otra forma.
6.  Verificar si los identificadores de producto de los eventos existen
    en el catálogo.

Consultas exploratorias orientativas (adaptar la sintaxis a RawTree y a
los ejemplos de `starter/queries.sql`):

``` sql
SELECT count() FROM shop.catalog;
SELECT count() FROM shop.history_events;
SELECT count() FROM shop.history_sessions;

SELECT * FROM shop.history_events LIMIT 10;
SELECT * FROM shop.history_sessions LIMIT 10;
SELECT * FROM shop.catalog LIMIT 10;
```

Entregable: una ficha breve con el tamaño de las tablas, los campos
disponibles, sus tipos observados y los problemas detectados.

## 1.2. Entender la distribución de los datos

Calcular, como mínimo:

-   Número de usuarios distintos.
-   Número de productos distintos con interacciones.
-   Número de eventos por tipo (`view`, `add_to_cart`, `purchase`).
-   Número de interacciones por usuario.
-   Número de interacciones por producto.
-   Número de usuarios por producto y productos por usuario.
-   Distribución temporal de eventos por día y hora.
-   Actividad por dispositivo, región y tipo de página cuando sea
    relevante.
-   Porcentaje de usuarios con una sola interacción.
-   Porcentaje de productos sin interacciones.

Visualizaciones recomendadas:

1.  Histograma de interacciones por usuario.
2.  Histograma de interacciones por producto.
3.  Evolución diaria de views, carts y purchases.
4.  Distribución de popularidad de productos.
5.  Matriz o tabla de densidad: usuarios, productos e interacciones.

Estas comprobaciones permiten saber si los datos son muy dispersos, si
unos pocos productos dominan el historial y cuántos usuarios tienen
información suficiente para calcular similitudes.

## 1.3. Decidir qué cuenta como interacción positiva

Como el problema es de recomendación, se puede empezar con feedback
implícito: no hay una puntuación explícita del usuario, sino acciones
observadas.

Una primera ponderación experimental podría ser:

  Evento            Peso inicial
  --------------- --------------
  `view`                       1
  `add_to_cart`                3
  `purchase`                   5

Estos pesos son **hipótesis iniciales**, no valores dados por el
challenge. Se deben probar varias configuraciones.

Recomendaciones:

-   Empezar con los pesos anteriores para tener una referencia sencilla.
-   Comparar después con, por ejemplo, `view=1`, `cart=2`, `purchase=4`.
-   Probar una variante que no use `view`, ya que una visita puede ser
    accidental.
-   Evitar que muchas visualizaciones repetidas de un mismo producto
    dominen la puntuación: se puede limitar el peso acumulado o usar una
    transformación como `log(1 + número_de_eventos)`.
-   Considerar una ventana temporal o decaimiento: una interacción
    reciente puede pesar más que una antigua.

**Precaución con las impresiones:** que un producto aparezca en `shown`
no significa que el usuario lo haya elegido. Tampoco se debe tratar
automáticamente todo producto no comprado como un rechazo. Las
impresiones y las no-interacciones son datos de exposición, no
necesariamente feedback negativo.

## 1.4. Crear un conjunto de interacciones usuario-producto

El resultado de esta etapa debe ser una tabla agregada con una fila por
pareja usuario-producto.

Esquema conceptual:

  --------------------------------------------------------------------------------------------------------
  user_id   product_id     view_count   cart_count   purchase_count   interaction_score last_interaction
  --------- ------------ ------------ ------------ ---------------- ------------------- ------------------
  u1        p1                      3            1                0                   6 ...

  u1        p2                      1            0                1                   6 ...

  u2        p1                      2            0                0                   2 ...
  --------------------------------------------------------------------------------------------------------

La puntuación se calcula con los pesos elegidos. La tabla puede incluir
también la fecha de última interacción para poder aplicar decaimiento
temporal.

Mantener dos versiones si es posible:

-   **Eventos orgánicos:** agregados desde `shop.history_events`.
-   **Feedback de carrusel:** derivado de `shop.history_sessions`,
    distinguiendo impresiones, clics y compras.

En la primera versión de los modelos, se recomienda entrenar
principalmente con eventos orgánicos y medir por separado qué aporta
incorporar el feedback de carrusel. Así se puede identificar si el sesgo
de exposición empeora o mejora el modelo.

## 1.5. Tratar el sesgo y las interacciones no observadas

En `shop.history_sessions`, el sistema anterior elegía qué productos
mostrar. Por tanto, los productos populares están sobrerrepresentados.

Acciones:

1.  No interpretar un producto no mostrado como un producto que el
    usuario no quiere.
2.  No convertir automáticamente todas las impresiones sin clic en
    negativos fuertes.
3.  Si se usan impresiones, conservar la posición y el `propensity`
    correspondiente.
4.  Evaluar una variante entrenada solo con eventos orgánicos y otra que
    añada feedback de carrusel.
5.  Documentar claramente qué datos se han usado en cada experimento.

El uso de `propensity` requiere cuidado: hay que confirmar cómo se
alinea cada probabilidad con cada producto y posición, y qué estimador
de corrección se va a aplicar. No basta con dividir indiscriminadamente
todos los registros por ese valor.

## 1.6. Preparar una evaluación temporal sin fuga de información

No hacer una división aleatoria simple de las filas: puede hacer que el
modelo aprenda del futuro y se evalúe sobre el pasado.

Protocolo recomendado:

1.  Ordenar los eventos por `ts`.
2.  Elegir un corte temporal.
3.  Entrenar con eventos anteriores al corte.
4.  Evaluar con interacciones posteriores al corte.
5.  Repetir el experimento con más de un corte temporal si el tiempo lo
    permite.

Para cada usuario, el modelo solo debe conocer las interacciones
disponibles antes del instante de recomendación. Los productos comprados
después del corte son los objetivos de evaluación, no datos de
entrenamiento.

Métricas iniciales:

-   **Recall@10:** proporción de productos relevantes posteriores que
    aparecen entre las diez recomendaciones.
-   **Precision@10:** proporción de las diez recomendaciones que son
    relevantes según la definición adoptada.
-   **NDCG@10:** mide la calidad del orden, dando más valor a los
    aciertos en posiciones altas.
-   **Coverage:** proporción del catálogo que el recomendador llega a
    recomendar.
-   **Conversion o compra esperada:** cuando se pueda estimar de forma
    comparable con el mecanismo del challenge.

Es necesario definir qué significa "relevante" en cada evaluación: por
ejemplo, compra posterior, o clic/compra posterior. No mezclar
definiciones entre experimentos.

## 1.7. Construir un baseline de comparación

Antes de evaluar User-CF e Item-CF, crear una referencia sencilla:

-   Los diez productos activos más populares por compras.
-   Si no hay suficientes compras, usar una popularidad combinada de
    views, carts y purchases.
-   Excluir productos inactivos o sin stock.
-   Mantener la misma ventana temporal y los mismos datos disponibles
    para todos los modelos.

El README define el baseline oficial como los best sellers disponibles
en stock, ordenados al inicio del stream. Para la evaluación offline, se
puede añadir un baseline dinámico, pero hay que identificarlo como una
comparación adicional y no confundirlo con el baseline oficial.

**Entregable de la fase común:** dataset de interacciones validado,
informe exploratorio, protocolo de evaluación temporal y baseline
reproducible.

------------------------------------------------------------------------

# 2. Variante A: User-Based Collaborative Filtering (User-CF)

## 2.1. Idea

User-CF busca usuarios con patrones de comportamiento parecidos al
usuario que acaba de llegar.

Si los usuarios A y B han interactuado con productos similares, y B ha
mostrado interés por un producto que A todavía no ha visto, ese producto
puede ser una buena recomendación para A.

## 2.2. Plan de acción

### Paso A1. Construir la matriz usuario-producto

Filas: usuarios.\
Columnas: productos.\
Valores: puntuación de interacción agregada.

Ejemplo ilustrativo:

  Usuario     p1   p2   p3   p4
  --------- ---- ---- ---- ----
  A            5    2    0    0
  B            4    3    5    0
  C            0    0    4    5
  D            5    0    3    1

Utilizar una matriz dispersa (*sparse matrix*), porque la mayoría de
usuarios no interactúa con la mayoría de productos.

### Paso A2. Calcular similitud entre usuarios

Cada usuario se representa mediante su vector de interacciones con todos
los productos.

Una opción inicial es la similitud del coseno:

\[ sim(u,v)= `\frac{\mathbf{x}_u\cdot\mathbf{x}_v}`{=tex}
{\|`\mathbf{x}`{=tex}\_u\|\|`\mathbf{x}`{=tex}\_v\|} \]

También se puede probar una similitud basada en intersección, como
Jaccard, si se convierten las interacciones en presencia/ausencia.

Decisión inicial recomendada: empezar con coseno sobre puntuaciones
ponderadas y después comparar con una versión binaria.

### Paso A3. Seleccionar vecinos

Para un usuario objetivo (u):

1.  Calcular o recuperar sus usuarios más similares.
2.  Seleccionar los (k) vecinos más cercanos.
3.  Excluir al propio usuario.
4.  Probar varios valores de (k), por ejemplo 20, 50 y 100.
5.  Considerar un umbral mínimo de similitud para evitar vecinos poco
    relacionados.

No es necesario almacenar todas las similitudes entre todos los
usuarios. Si hay muchos usuarios, esa matriz puede ser demasiado grande.
Es preferible almacenar solo los vecinos más cercanos por usuario o
utilizar una búsqueda eficiente de vecinos.

### Paso A4. Puntuar productos candidatos

Una fórmula inicial es:

\[ Score(u,i)= `\frac{
\sum_{v\in N_k(u)}sim(u,v)\,r_{v,i}
}{
\sum_{v\in N_k(u)}|sim(u,v)|
}`{=tex} \]

Donde:

-   (N_k(u)) es el conjunto de los (k) usuarios vecinos.
-   (sim(u,v)) es la similitud entre el usuario objetivo y el vecino.
-   (r\_{v,i}) es la interacción del vecino (v) con el producto (i).

Se pueden probar variantes que descuenten la popularidad general del
producto o que limiten la contribución de usuarios muy activos.

### Paso A5. Filtrar y ordenar

Para la solicitud real:

1.  Obtener el usuario y su contexto.
2.  Recuperar los vecinos disponibles.
3.  Obtener productos con buena puntuación entre esos vecinos.
4.  Excluir productos ya vistos o comprados si la política del
    recomendador lo considera adecuado.
5.  Excluir productos inactivos o sin stock.
6.  Ordenar por puntuación.
7.  Devolver exactamente diez productos distintos.

### Paso A6. Tratar usuarios nuevos o con poco historial

User-CF necesita conocer el comportamiento del usuario para encontrar
vecinos.

Casos:

-   Usuario con historial suficiente: aplicar User-CF.
-   Usuario con pocas interacciones: usar los vecinos disponibles con
    cautela o combinar con popularidad.
-   Usuario nuevo sin interacciones: no hay base para encontrar vecinos;
    utilizar fallback de popularidad o una estrategia de
    contenido/contexto.

### Paso A7. Evaluar y ajustar

Experimentos:

-   Coseno con pesos de evento diferentes.
-   Coseno binario frente a ponderado.
-   Diferentes valores de (k).
-   Con y sin decaimiento temporal.
-   Con y sin eventos de carrusel.
-   Diferentes reglas para excluir productos ya vistos.

Registrar las métricas Recall@10, Precision@10, NDCG@10, cobertura y
tiempo de generación.

## 2.3. Ventajas y limitaciones de User-CF

**Ventajas**

-   Personaliza en función de gustos de otros usuarios.
-   Puede descubrir productos distintos de los que el usuario ya ha
    consultado.
-   Puede aprovechar patrones colectivos que no son evidentes por
    categoría o marca.

**Limitaciones**

-   Usuarios con pocas interacciones son difíciles de comparar.
-   La similitud entre usuarios puede ser costosa si hay muchos
    usuarios.
-   Los gustos pueden cambiar con el tiempo.
-   Si la matriz es muy dispersa, muchos usuarios tendrán pocos
    productos en común.

**Riesgo técnico principal:** escalar el cálculo y almacenamiento de
vecinos. Para el challenge, hay que precalcular o actualizar vecinos de
forma controlada; no calcular similitudes contra todos los usuarios en
cada solicitud.

------------------------------------------------------------------------

# 3. Variante B: Item-Based Collaborative Filtering (Item-CF)

## 3.1. Idea

Item-CF busca productos que suelen recibir interacciones de los mismos
usuarios.

Si muchos usuarios que han visto o comprado el producto A también han
interactuado con el producto B, ambos productos pueden estar
relacionados.

Cuando llega una solicitud, se recomiendan productos similares a los
productos que el usuario ya ha consultado.

## 3.2. Plan de acción

### Paso B1. Construir la matriz usuario-producto

Se utiliza la misma matriz de interacciones de la fase común:

-   Filas: usuarios.
-   Columnas: productos.
-   Valores: pesos o puntuaciones de interacción.

Para Item-CF, cada columna representa el patrón de usuarios que han
interactuado con ese producto.

### Paso B2. Calcular similitud entre productos

Representar cada producto mediante su vector de interacciones con todos
los usuarios.

Aplicar similitud del coseno:

\[ sim(i,j)= `\frac{\mathbf{x}_i\cdot\mathbf{x}_j}`{=tex}
{\|`\mathbf{x}`{=tex}\_i\|\|`\mathbf{x}`{=tex}\_j\|} \]

Aquí (`\mathbf{x}`{=tex}\_i) y (`\mathbf{x}`{=tex}\_j) son las columnas
correspondientes a los productos (i) y (j).

Probar también una versión binaria para comprobar si los pesos de las
interacciones aportan valor.

### Paso B3. Guardar los vecinos de cada producto

Para cada producto, conservar únicamente sus (k) productos más
similares.

Ejemplo de estructura:

  product_id   similar_product_id     similarity
  ------------ -------------------- ------------
  p1           p8                           0.91
  p1           p4                           0.76
  p1           p7                           0.62

Probar distintos tamaños de vecindario, por ejemplo 20, 50 y 100.

Esta tabla de vecinos permite responder rápidamente sin recalcular la
matriz completa en cada petición.

### Paso B4. Puntuar productos candidatos

Para un usuario (u), sea (I_u) el conjunto de productos con los que ha
interactuado.

Una fórmula inicial:

\[ Score(u,j)= `\frac{
\sum_{i\in I_u}r_{u,i}\,sim(i,j)
}{
\sum_{i\in I_u}|r_{u,i}|
}`{=tex} \]

Donde:

-   (r\_{u,i}) es la fuerza de la interacción del usuario con el
    producto (i).
-   (sim(i,j)) es la similitud entre el producto conocido (i) y el
    candidato (j).

Para hacerlo eficiente:

1.  Recuperar los últimos productos relevantes del usuario.
2.  Consultar los vecinos precalculados de esos productos.
3.  Acumular puntuaciones de candidatos.
4.  Eliminar duplicados y productos no válidos.
5.  Ordenar y seleccionar diez.

### Paso B5. Tratar usuarios nuevos y productos nuevos

-   Usuario sin historial: Item-CF no tiene productos de partida; usar
    fallback.
-   Producto nuevo sin interacciones: no tendrá vecinos colaborativos
    fiables.
-   Producto que vuelve a estar disponible: puede reincorporarse si
    conserva interacciones históricas y está activo en el catálogo
    actual.

Para productos realmente nuevos, el contenido del catálogo (categoría,
subcategoría, marca, precio y descripción) puede ayudar, pero eso ya
sería una extensión basada en contenido, no Collaborative Filtering
puro.

### Paso B6. Evaluar y ajustar

Experimentos:

-   Diferentes pesos para view, cart y purchase.
-   Coseno ponderado y binario.
-   Diferentes valores de (k).
-   Diferentes ventanas temporales.
-   Con y sin decaimiento temporal.
-   Con y sin eventos de carrusel.
-   Diferentes cantidades de productos recientes del usuario usados como
    punto de partida.

Medir las mismas métricas y utilizar exactamente los mismos cortes
temporales que en User-CF.

## 3.3. Ventajas y limitaciones de Item-CF

**Ventajas**

-   Los productos suelen cambiar menos que los usuarios.
-   Se pueden precalcular vecinos de productos.
-   La inferencia puede ser rápida: basta con consultar vecinos de los
    productos recientes del usuario.
-   Es fácil de explicar y depurar.

**Limitaciones**

-   Un usuario nuevo sin historial sigue siendo un caso difícil.
-   Los productos nuevos no tienen suficientes interacciones para
    calcular similitudes.
-   Puede recomendar productos muy parecidos y reducir la diversidad.
-   Los productos muy populares pueden dominar las relaciones.

**Riesgo técnico principal:** calcular y mantener una tabla de vecinos
de productos que sea suficientemente buena y rápida, sin consumir
demasiada memoria.

------------------------------------------------------------------------

# 4. Variante C: modelo combinado User-CF + Item-CF

## 4.1. Idea

En esta tercera variante se ejecutan los dos modelos de forma
independiente y después se combinan sus puntuaciones.

-   User-CF aporta productos que han interesado a usuarios con gustos
    parecidos.
-   Item-CF aporta productos relacionados con los productos que el
    usuario ya ha consultado.

La combinación puede aprovechar las ventajas de ambos métodos y
compensar algunas de sus limitaciones.

**Importante:** primero se deben entrenar, evaluar y entender los dos
modelos por separado. El modelo combinado no sustituye esa comparación:
es un tercer experimento.

## 4.2. Plan de acción

### Paso C1. Mantener dos modelos independientes

Conservar:

-   Matriz de interacciones compartida.
-   Índice de usuarios y productos.
-   Vecinos de usuarios para User-CF.
-   Vecinos de productos para Item-CF.
-   Parámetros y métricas de cada modelo.

Aunque ambos parten de la misma tabla de interacciones, sus similitudes
y sus procedimientos de recomendación son diferentes.

### Paso C2. Generar candidatos con ambos modelos

Para una solicitud:

1.  User-CF genera una lista de candidatos y sus puntuaciones.
2.  Item-CF genera otra lista de candidatos y sus puntuaciones.
3.  Unir las dos listas.
4.  Eliminar duplicados.
5.  Conservar la puntuación de cada modelo para cada candidato.

No es obligatorio que ambos modelos propongan los mismos productos.

### Paso C3. Normalizar las puntuaciones

Las puntuaciones de User-CF e Item-CF pueden tener escalas diferentes.
No conviene sumarlas directamente sin comprobarlo.

Opciones:

-   Normalización Min-Max por solicitud.
-   Normalización por rango.
-   Normalización mediante percentiles.

Una opción sencilla para empezar es Min-Max:

\[ s'=`\frac{s-s_{\min}}{s_{\max}-s_{\min}}`{=tex} \]

Si todas las puntuaciones son iguales, definir un tratamiento específico
para evitar división por cero.

La normalización debe hacerse sobre los candidatos de cada modelo para
la misma solicitud.

### Paso C4. Combinar las puntuaciones

Una primera combinación lineal:

\[ Score\_{`\text{hybrid}`{=tex}}(u,i)=
`\alpha`{=tex},Score'*{`\text{UserCF}`{=tex}}(u,i)
+(1-`\alpha`{=tex}),Score'*{`\text{ItemCF}`{=tex}}(u,i) \]

Donde:

-   (Score'\_{`\text{UserCF}`{=tex}}) es la puntuación normalizada de
    User-CF.
-   (Score'\_{`\text{ItemCF}`{=tex}}) es la puntuación normalizada de
    Item-CF.
-   (`\alpha`{=tex}) controla la importancia de User-CF.

Probar, por ejemplo:

-   (`\alpha=0`{=tex}): solo Item-CF.
-   (`\alpha=0.25`{=tex}).
-   (`\alpha=0.5`{=tex}): ambos tienen el mismo peso.
-   (`\alpha=0.75`{=tex}).
-   (`\alpha=1`{=tex}): solo User-CF.

Los extremos deben reproducir los resultados de los modelos
individuales, siempre que se mantengan iguales el conjunto de candidatos
y las reglas de filtrado.

### Paso C5. Añadir reglas según la disponibilidad de información

La combinación no tiene por qué utilizar siempre el mismo peso.

Ejemplos de reglas que se pueden probar:

-   Si el usuario tiene historial suficiente y hay vecinos fiables, dar
    más peso a User-CF.
-   Si el usuario tiene varios productos recientes y Item-CF tiene
    vecinos fiables, dar más peso a Item-CF.
-   Si User-CF no puede generar candidatos, utilizar Item-CF.
-   Si Item-CF no puede generar candidatos, utilizar User-CF.
-   Si ambos fallan, utilizar el fallback de popularidad.

Estas reglas deben definirse antes de evaluar y compararse con la
combinación de peso fijo.

### Paso C6. Diversidad y filtrado final

Después de combinar:

1.  Eliminar productos inactivos o sin stock.
2.  Aplicar la política de exclusión de productos ya vistos/comprados.
3.  Evitar duplicados.
4.  Seleccionar los diez primeros.
5.  Si faltan candidatos, completar con un fallback válido.

La diversidad se puede estudiar como experimento adicional: por ejemplo,
limitar recomendaciones excesivamente redundantes. No conviene
introducir demasiadas reglas al principio, porque dificultaría saber qué
mejora realmente el resultado.

### Paso C7. Evaluar el modelo híbrido

Comparar:

-   User-CF solo.
-   Item-CF solo.
-   Híbrido con (`\alpha=0.25`{=tex}).
-   Híbrido con (`\alpha=0.5`{=tex}).
-   Híbrido con (`\alpha=0.75`{=tex}).
-   Híbrido con pesos adaptativos.

Usar los mismos datos, cortes temporales, filtros y métricas.

La combinación solo merece la pena si mejora la calidad de las
recomendaciones lo suficiente como para compensar el coste adicional de
cálculo y mantenimiento.

## 4.3. Ventajas y limitaciones del modelo combinado

**Ventajas**

-   Combina dos fuentes distintas de señal colaborativa.
-   Puede mejorar la cobertura de candidatos.
-   Permite adaptarse a situaciones en las que uno de los dos modelos
    tiene más información.
-   Es una forma razonable de construir un recomendador más robusto.

**Limitaciones**

-   Más complejidad de implementación y depuración.
-   Más parámetros que ajustar.
-   Puede aumentar la latencia.
-   Una mala normalización puede hacer que un modelo domine al otro.
-   No resuelve por sí solo el cold start de usuarios y productos.

------------------------------------------------------------------------

# 5. Comparación de los tres experimentos

  -----------------------------------------------------------------------
  Aspecto           User-CF           Item-CF           Híbrido
  ----------------- ----------------- ----------------- -----------------
  Busca similitud   Usuarios          Productos         Ambas
  entre                                                 

  Señal principal   Gustos de         Productos         Combinación de
                    usuarios          relacionados      ambas
                    parecidos                           

  Cálculo que       Vecinos de        Vecinos de        Ambos
  conviene          usuarios          productos         
  precalcular                                           

  Usuario nuevo     Problema          Problema          Sigue siendo un
                    importante        importante        problema

  Producto nuevo    Puede aparecer en No tiene vecinos  Sigue siendo un
                    vecinos solo si   fiables           problema
                    tiene                               
                    interacciones                       

  Complejidad       Media/alta según  Media según       Mayor
                    usuarios          productos         

  Objetivo          Medir valor de    Medir valor de    Comprobar si se
  experimental      vecinos de        vecinos de        complementan
                    usuarios          productos         
  -----------------------------------------------------------------------

No se puede afirmar de antemano que una variante vaya a ganar. Dependerá
de la dispersión de los datos, del número de interacciones por
usuario/producto, de la distribución temporal y de cómo se evalúen las
recomendaciones.

------------------------------------------------------------------------

# 6. Diseño de experimentos: cómo comparar de forma justa

Para que la comparación sea válida, las tres variantes deben utilizar:

-   El mismo periodo de entrenamiento.
-   Los mismos periodos de validación.
-   La misma definición de interacción relevante.
-   Los mismos pesos de evento en la primera comparación.
-   Los mismos filtros de productos activos.
-   La misma política de productos ya vistos.
-   El mismo límite de diez recomendaciones.
-   El mismo baseline.

Después de esta comparación inicial, se pueden optimizar los parámetros
de cada modelo de manera independiente.

## Tabla de resultados que conviene rellenar

  --------------------------------------------------------------------------------
  Experimento      Recall@10   Precision@10      NDCG@10     Coverage     Latencia
  ------------- ------------ -------------- ------------ ------------ ------------
  Baseline                                                            

  User-CF                                                             

  Item-CF                                                             

  Híbrido 25/75                                                       

  Híbrido 50/50                                                       

  Híbrido 75/25                                                       

  Híbrido                                                             
  adaptativo                                                          
  --------------------------------------------------------------------------------

Añadir también:

-   Número de solicitudes evaluadas.
-   Número de usuarios con historial suficiente.
-   Número de productos sin interacciones.
-   Porcentaje de solicitudes con diez recomendaciones válidas.
-   Tiempo de construcción/actualización del modelo.
-   Memoria utilizada.

No elegir el ganador mirando una única métrica. Para este challenge, la
conversión esperada, la fiabilidad y la capacidad de responder a
solicitudes nuevas son especialmente importantes.

------------------------------------------------------------------------

# 7. Arquitectura de implementación recomendada

Una estructura de proyecto posible:

``` text
project/
│
├── data/
│   ├── exploratory_analysis.py
│   └── build_interactions.py
│
├── models/
│   ├── user_cf.py
│   ├── item_cf.py
│   └── hybrid_cf.py
│
├── evaluation/
│   ├── temporal_split.py
│   ├── metrics.py
│   └── compare_models.py
│
├── artifacts/
│   ├── user_neighbors/
│   ├── item_neighbors/
│   └── model_config.json
│
├── recommender.py
└── README.md
```

Esta es una propuesta de organización, no una estructura exigida por el
challenge. El starter kit ya proporciona `starter/recommender.py`,
`starter/queries.sql` y `starter/offline_eval.ipynb`; se debe integrar
la solución respetando el flujo y las interfaces existentes.

## Flujo offline

1.  Consultar datos históricos.
2.  Validar y agregar interacciones.
3.  Crear el corte temporal.
4.  Construir matrices dispersas.
5.  Entrenar User-CF.
6.  Entrenar Item-CF.
7.  Generar predicciones en validación.
8.  Entrenar/probar el modelo híbrido.
9.  Calcular métricas.
10. Guardar configuraciones y resultados.

## Flujo online

1.  Leer solicitudes nuevas.
2.  Consultar el estado actual del catálogo y aplicar actualizaciones.
3.  Recuperar los datos recientes del usuario.
4.  Ejecutar el recomendador seleccionado.
5.  Filtrar y validar diez productos.
6.  Insertar la respuesta en `team_NN.submissions`.
7.  Leer `team_NN.feedback` para medir el rendimiento.
8.  Actualizar estadísticas/modelos de forma controlada.

**Restricción crítica del challenge:** cada respuesta debe llegar en
menos de tres segundos desde el timestamp de la solicitud. El README
recomienda agrupar inserciones (por ejemplo, cada medio segundo) en
lugar de realizar una llamada HTTP por solicitud. No se debe bloquear la
generación de recomendaciones esperando a recalcular matrices completas.

------------------------------------------------------------------------

# 8. Orden de trabajo recomendado

La siguiente secuencia reduce el riesgo de invertir tiempo en un modelo
complejo antes de saber si los datos sirven.

## Fase 1 --- Exploración

-   [ ] Acceder a las tres tablas históricas.
-   [ ] Medir tamaños y distribuciones.
-   [ ] Analizar la dispersión usuario-producto.
-   [ ] Revisar calidad y consistencia de IDs.
-   [ ] Construir las primeras tablas agregadas de interacciones.
-   [ ] Definir pesos y protocolo temporal.
-   [ ] Ejecutar el baseline offline.

**Criterio para avanzar:** poder construir una tabla usuario-producto
reproducible y explicar qué representa cada valor.

## Fase 2 --- User-CF independiente

-   [ ] Construir la matriz dispersa.
-   [ ] Calcular similitudes entre usuarios.
-   [ ] Extraer vecinos.
-   [ ] Puntuar productos.
-   [ ] Generar diez recomendaciones.
-   [ ] Evaluar con los cortes temporales.
-   [ ] Registrar latencia y cobertura.

**Criterio para avanzar:** tener resultados reproducibles y una
comparación clara con el baseline.

## Fase 3 --- Item-CF independiente

-   [ ] Reutilizar la matriz de interacciones.
-   [ ] Calcular similitudes entre productos.
-   [ ] Guardar vecinos de productos.
-   [ ] Puntuar candidatos desde el historial del usuario.
-   [ ] Generar diez recomendaciones.
-   [ ] Evaluar con los mismos cortes.
-   [ ] Registrar latencia y cobertura.

**Criterio para avanzar:** conocer la calidad y el coste de Item-CF
frente a User-CF.

## Fase 4 --- Híbrido

-   [ ] Mantener ambos modelos independientes.
-   [ ] Generar candidatos de ambos.
-   [ ] Normalizar puntuaciones.
-   [ ] Probar distintos valores de alpha.
-   [ ] Probar reglas adaptativas.
-   [ ] Evaluar con el mismo protocolo.
-   [ ] Comprobar si la mejora compensa la latencia.

**Criterio para avanzar:** demostrar que la combinación mejora una
métrica relevante sin perjudicar la fiabilidad.

## Fase 5 --- Integración y competición

-   [ ] Integrar el modelo ganador en el starter.
-   [ ] Asegurar exactamente diez IDs distintos y activos.
-   [ ] Comprobar la latencia real de extremo a extremo.
-   [ ] Preparar fallback para usuarios sin historial.
-   [ ] Incorporar cambios de catálogo.
-   [ ] Leer feedback y monitorizar resultados.
-   [ ] Mantener el proceso funcionando sin que el equipo tenga que
    intervenir manualmente.

------------------------------------------------------------------------

# 9. Decisiones que conviene tomar antes de programar

Hay algunas decisiones que no están resueltas por el README y que deben
quedar documentadas:

1.  **Fuente principal de entrenamiento:** empezar con
    `shop.history_events`; después medir si añadir feedback de
    `shop.history_sessions` mejora el resultado.
2.  **Pesos de eventos:** empezar con 1/3/5 y probar alternativas.
3.  **Ventana temporal:** comparar los 60 días completos con ventanas
    más recientes.
4.  **Decaimiento temporal:** comprobar si dar más peso a eventos
    recientes mejora las predicciones.
5.  **Interacciones repetidas:** decidir si se suman, se limitan o se
    transforman.
6.  **Productos ya vistos:** definir si se excluyen siempre o si se
    pueden volver a recomendar cuando existe una señal fuerte.
7.  **Productos comprados:** decidir si se excluyen para evitar
    recomendar una compra ya realizada, teniendo en cuenta que el
    challenge no especifica una regla obligatoria.
8.  **Cold start:** diseñar un fallback independiente de CF.
9.  **Actualización online:** decidir cada cuánto se actualizan vecinos
    y popularidad sin poner en riesgo los tres segundos.
10. **Criterio de selección:** priorizar la métrica de conversión
    esperada del challenge, sin descuidar fiabilidad y cold start.

------------------------------------------------------------------------

# 10. Recomendación final

Mi recomendación es desarrollar las tres variantes, pero **no
simultáneamente desde el primer día**.

El orden más sensato es:

1.  Crear y validar una única base de interacciones.
2.  Implementar User-CF y medirlo.
3.  Implementar Item-CF y medirlo.
4.  Comparar ambos y entender en qué casos falla cada uno.
5.  Construir el híbrido usando los resultados de los dos modelos.
6.  Elegir la alternativa que ofrezca el mejor equilibrio entre
    conversión, cobertura, latencia y fiabilidad.

La pregunta central no es solamente "¿qué algoritmo obtiene el mejor
Recall@10?", sino:

**¿Qué información aporta User-CF que Item-CF no aporta, y viceversa?
¿Esa información adicional consigue que el cliente compre más
productos?**

Si el equipo consigue responder experimentalmente a esa pregunta, tendrá
una justificación sólida tanto para el sistema final como para la
presentación ante el jurado.
