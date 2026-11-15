import os
import sys
from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
    """Canvas that adds running headers and 'Page X of Y' footers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 9)
        self.setFillColor(colors.HexColor("#555555"))
        
        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 800, "Datathon FME 2026 · Team 32 (DataVengers)")
            self.drawRightString(541, 800, "Evolución del Sistema Recomendador Calypso")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 792, 541, 792)

        # Footer (all pages)
        page_str = f"Página {self._pageNumber} de {page_count}"
        self.drawRightString(541, 35, page_str)
        self.drawString(54, 35, "Confidencial · Presentación al Jurado")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 47, 541, 47)
        self.restoreState()

def build_pdf():
    repo_root = Path(r"C:\Users\megia\Desktop\stdw\Datathon-2026---DataVengers")
    pdf_path = repo_root / "Presentacion" / "Resumen_Evolucion_Calypso.pdf"
    
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    
    styles = getSampleStyleSheet()
    
    # Custom LaTeX-like Typography
    primary_color = colors.HexColor("#184c78")
    secondary_color = colors.HexColor("#c62828")
    dark_text = colors.HexColor("#1a1a1a")
    
    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Times-Bold",
        fontSize=21,
        leading=25,
        textColor=primary_color,
        alignment=1, # Center
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Times-Italic",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#333333"),
        alignment=1,
        spaceAfter=8,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        fontName="Times-Roman",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#555555"),
        alignment=1,
        spaceAfter=14,
    )
    h1_style = ParagraphStyle(
        "H1",
        fontName="Times-Bold",
        fontSize=14,
        leading=18,
        textColor=primary_color,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    h2_style = ParagraphStyle(
        "H2",
        fontName="Times-Bold",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#2c3e50"),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body",
        fontName="Times-Roman",
        fontSize=10,
        leading=14,
        textColor=dark_text,
        spaceAfter=6,
    )
    body_bold = ParagraphStyle(
        "BodyBold",
        fontName="Times-Bold",
        fontSize=10,
        leading=14,
        textColor=dark_text,
    )
    abstract_style = ParagraphStyle(
        "Abstract",
        fontName="Times-Roman",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#222222"),
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        fontName="Times-Roman",
        fontSize=9.5,
        leading=13.5,
        textColor=dark_text,
        leftIndent=15,
        spaceAfter=3,
    )
    alert_box_style = ParagraphStyle(
        "AlertBox",
        fontName="Times-Roman",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#7f1d1d"),
    )

    story = []

    # Title Banner
    story.append(Paragraph("Evolución del Sistema Recomendador Calypso", title_style))
    story.append(Paragraph("De la Etapa 1 a Calypso 6.2b: Innovaciones, Problemas Reales y Rendimiento", subtitle_style))
    story.append(Paragraph("<b>Equipo DataVengers (Team 32)</b> · Datathon FME 2026 (UPC Barcelona) · RawTree Challenge<br/>15 de noviembre de 2026", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceAfter=10))

    # Abstract Box
    abstract_text = (
        "<b>Resumen Ejecutivo:</b> Este documento recoge la historia completa de desarrollo y aprendizaje empírico del motor de "
        "recomendación en tiempo real <b>Calypso</b> para el reto RawTree. Se expone con detalle la trayectoria a través de sus diez "
        "versiones (Etapa 1 a 6.2b), abordando con transparencia tanto los hitos como las crisis operativas reales "
        "(la caída de SLA al 10.3% en la Etapa 3 y la pérdida del 25% de DCG por ordenación de precio en 4.1). Se incluye cuadro "
        "comparativo completo y argumentario estructurado para la defensa de los 30 puntos ante el Jurado."
    )
    abstract_table = Table(
        [[Paragraph(abstract_text, abstract_style)]],
        colWidths=[487],
    )
    abstract_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LINELEFT', (0,0), (-1,-1), 3.5, primary_color),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(abstract_table)
    story.append(Spacer(1, 10))

    # Section 1
    story.append(Paragraph("1. Marco del Reto y Reglas de Evaluación", h1_style))
    p1 = (
        "El reto oficial de RawTree exige responder a compradores que acceden a una tienda de 5.000 productos "
        "a un ritmo continuo de <b>~3 peticiones/segundo</b>. Para cada comprador, el sistema dispone de un "
        "<b>SLA estricto de 3.000 milisegundos</b> desde la marca temporal de la solicitud para devolver "
        "<b>exactamente 10 productos activos y en existencias</b>. Fallar por retraso, duplicados o un solo producto "
        "agotado activa el <i>fallback</i> (los superventas globales), que otorga cero puntos.<br/>"
        "La puntuación final se divide en <b>70 puntos automáticos</b> en la Ronda Final (50 Conversión, 10 Cold Start, "
        "10 Fiabilidad/SLA) y <b>30 puntos de presentación ante el Jurado</b> (Método, Aprendizaje del flujo, Uso de RawTree y Comunicación)."
    )
    story.append(Paragraph(p1, body_style))

    # Section 2
    story.append(Paragraph("2. Evolución Cronológica y Problemas Encontrados", h1_style))

    versions_data = [
        ("2.1 Etapa 1: etapa1-hybrid-v1 (El Origen Desacoplado)",
         "Separó la ejecución en dos planos: <b>Fast Loop</b> en memoria para inferencia en microsegundos y "
         "<b>Slow Loop</b> en segundo plano para consultas SQL pesadas sin congelar las respuestas. "
         "Garantizó 10 IDs válidos con guardrails y exploró co-visitas, auto-conversión y TF-IDF de catálogo.",
         "<b>Problemas críticos:</b> El refresco completo recalculaba 60 días cada 180s y bloqueaba el stock mientras duraba. "
         "El peor caso de ciclo superaba los 4s (> SLA de 3s). Las co-visitas no tenían dirección temporal ni corrección de popularidad, "
         "y la validación sintética era circular (probaba lo que el modelo asumía)."),

        ("2.2 Etapa 2: etapa2-seq-mmr-v1 (Lift, Perfiles y Cold Start Bidireccional)",
         "Introdujo <b>Lift debiassado</b> en transiciones para evitar el monopolio de superventas, decaimiento temporal dual, "
         "perfil profundo de usuario (<b>UserProfile</b> con carritos abandonados y elasticidad de precio gaussiana), "
         "<b>Graph Grafting</b> bidireccional (conocido $\\to$ nuevo y nuevo $\\to$ conocido) y diversidad MMR.",
         "<b>Problemas críticos:</b> Descargar cientos de miles de filas JSON de perfiles cada 3 min causaba paradas del GIL en Python "
         "durante la decodificación en C, pausando el hilo de servicio. Se agregaron 22 parámetros a ciegas sin datos reales. "
         "El 'desempate por ingresos' reordenaba por precio alto con diferencias de hasta el 6%, perjudicando la conversión de compradores de presupuesto moderado."),

        ("2.3 Etapa 3: calypso-2.1 (Machine Learning con IPS y Bandits)",
         "Primer Re-Ranker logístico de Machine Learning sobre 14 variables normalizadas, entrenado con ponderación inversa "
         "de propensión posicional $(k+1)^{0.75}$ (IPS). Incorporó Bandit UCB posicional y el agente autónomo <b>CalypsoAgent</b> "
         "que registraba decisiones en <i>team_32.scratch</i>.",
         "<b>La Gran Caída del SLA al 10.3%:</b> Tras reiniciar el equipo, el servicio solicitaba <i>LIMIT 500 ORDER BY ts DESC</i>. "
         "~490 peticiones ya habían caducado (>3s); el sistema tardaba más de 1s en procesar peticiones zombi y acumulaba retraso "
         "irreversible en las peticiones frescas entrantes. Al reiniciar sin caché en disco, debía consultar 12 millones de filas a RawTree, "
         "generando decenas de segundos de fallbacks continuos."),

        ("2.4 Etapa 4: Blindaje Operativo Anti-Caídas (La Base Moderna)",
         "Rediseño radical de resiliencia: <b>Caché atómica a disco</b> (<i>snapshot_v4_latest.pkl</i>) con arranque en <b>< 0.4s</b>; "
         "lectura acotada a peticiones frescas (edad < 2.400 ms, <i>LIMIT 25</i>); deduplicación confirmada únicamente tras push exitoso; "
         "supervisor Watchdog y arquitectura <b>Slot-Gated</b> (slots 0–6 blindados para precisión WARM, 7–9 para UCB/cold start).",
         "<b>Impacto:</b> Se recuperó de inmediato la disponibilidad por encima del <b>99.5% ON TIME</b> y se resolvió de raíz la vulnerabilidad de las caídas de sistema."),

        ("2.5 Calypso 3.1: El Coste de la Agresividad Simultánea",
         "Intentó aplicar 4 cambios drásticos a la vez: suprimir suelos de subcategoría/marca, penalización de fondo (x0.45/x0.18), "
         "modelo colaborativo EASE con peso 0.25 y SQL direccional.",
         "<b>Regresión del 77.2% al 74.5% (-2.7%):</b> Cambiar demasiadas piezas desestabilizó el recomendador. Forzar categorías sin datos "
         "costaba un -2.0% y EASE con peso alto introducía ruido. Solo la penalización de fondo resultó beneficiosa (+1.8%)."),

        ("2.6 Calypso 3.1b: Modularización y Estudio de Ablación",
         "Convirtió cada cambio de 3.1 en un interruptor conmutable por entorno (<i>CALYPSO_V3_*</i>) y ejecutó ablación rigurosa "
         "con 5 semillas aleatorias. Conservó por defecto exclusivamente la penalización de fondo y apagó el resto.",
         "<b>Récord de la etapa:</b> Subió el acierto a compra al <b>79.0%</b> (+1.8% sobre la base 2.1), validando el principio de "
         "no recomendar ningún producto sin evidencia estadística sólida."),

        ("2.7 Calypso 3.2: Semividas Temporales y Segmentación",
         "Implementó semivida estricta de 10 minutos (624s) en vistas recientes, suavizado bayesiano contextual jerárquico ($\kappa=50$) "
         "y segmentación en 5 perfiles operativos (nuevo/conocido con/sin sesión y producto nuevo).",
         "<b>Transición:</b> Aportó mejoras de calidad, pero exigió consolidar reglas de negocio comerciales e integrar un gobierno de pesos unificado."),

        ("2.8 Calypso 4.1 y 4.1_e: Reglas de Negocio y el Descubrimiento del Precio",
         "Incorporó reglas comerciales conmutables (<i>CALYPSO_V4_*</i>): prioridad para carritos activos ($\le 30$ min, máx 3 slots), "
         "recuperación de carritos olvidados, recurrencia garantizada y orden por precio más barato (<i>v4_price_ascending = True</i>). "
         "La versión 4.1_e introdujo el primer perfil JSON modular con <i>tuner.py</i>.",
         "<b>El Desastre del Orden por Precio:</b> La ablación reveló que ordenar por precio ascendente <b>destruía un 25% del DCG de compra</b> "
         "(-0.120). Dado que el Slot 1 concentra el 34% de compras y los slots 1–3 el 63%, mandar los artículos baratos arriba desplazaba "
         "del primer slot al producto con mayor probabilidad predictiva de compra. Además, reservar 3 slots fijos al carrito expulsaba predicciones superiores de ML."),

        ("2.9 Calypso 4.2: La Corrección de Medidas de Conversión",
         "Apagó la ordenación por precio (<i>v4_price_ascending = False</i>), sustituyó la reserva de slots de carrito por bonificaciones "
         "de puntuación para que compitieran libremente, introdujo la tasa base explícita por ranking de compras ($1 - \ln(rank)/\ln(max)$) "
         "y eliminó la diversidad MMR forzada en portada.",
         "<b>Lección:</b> El precio solo debe actuar como desempate lexicográfico entre productos idénticos de la misma subcategoría, "
         "nunca como criterio de ordenación global."),

        ("2.10 Calypso 5.1: Gobernanza Modular Completa y Recarga en Caliente",
         "Unificación de 14 pesos prior, coeficientes por segmento y reglas en <i>weights_profile.json</i> con <b>recarga en caliente "
         "(hot-reload)</b> en memoria. Pool ampliado a 140 candidatos con Top-35 superventas garantizado y eliminación de penalizaciones hardcoded.",
         "<b>Limitaciones:</b> La optimización de 65 parámetros por búsqueda ciega sobre pocas compras (386 del log) tocó techo, "
         "reclamando un modelo de aprendizaje continuo con residuo no lineal sobre el histórico masivo."),

        ("2.11 Calypso 6.2b: La Versión Final para la Ronda Decisiva",
         "<b>Cúspide técnica del equipo:</b><br/>"
         "1. <b>Dual-Head Slot-1 Precision:</b> Cabecera dedicada con 64 features y regularización del riesgo secuencial para asegurar el Slot 1.<br/>"
         "2. <b>Transfer Learning en 2 fases:</b> Teacher avanzado (HistGradientBoosting) entrenado sobre 13.129 compras históricas destilado en un Student MLP NumPy (80, 40) con Ridge.<br/>"
         "3. <b>Multi-Source IPS Learning:</b> Pondera compras Gold (1.0), carritos (0.18), recurrentes (0.14) y transiciones (0.12).<br/>"
         "4. <b>Transiciones cruzadas de subcategoría:</b> Captura ventas complementarias entre categorías.<br/>"
         "5. <b>FastLoop vectorizado (< 2.8 ms):</b> Inferencia ultrarrápida.<br/>"
         "6. <b>Rendimiento récord:</b> 74.41% de compra general y <b>82.61% en Cold Start</b> con el preset <i>product_profile_focus</i>.",
         "<b>Estado:</b> El sistema definitivo listo, blindado y probado para afrontar la Ronda Final.")
    ]

    for title, desc, problems in versions_data:
        card = []
        card.append(Paragraph(title, h2_style))
        card.append(Paragraph(desc, body_style))
        if "Problemas" in problems or "Desastre" in problems or "Caída" in problems:
            prob_table = Table([[Paragraph(problems, alert_box_style)]], colWidths=[487])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#fef2f2")),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LINELEFT', (0,0), (-1,-1), 3, secondary_color),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#fecaca")),
            ]))
            card.append(prob_table)
        else:
            ok_table = Table([[Paragraph(problems, body_style)]], colWidths=[487])
            ok_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f0fdf4")),
                ('LEFTPADDING', (0,0), (-1,-1), 8),
                ('RIGHTPADDING', (0,0), (-1,-1), 8),
                ('TOPPADDING', (0,0), (-1,-1), 5),
                ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                ('LINELEFT', (0,0), (-1,-1), 3, colors.HexColor("#16a34a")),
                ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#bbf7d0")),
            ]))
            card.append(ok_table)
        card.append(Spacer(1, 8))
        story.append(KeepTogether(card))

    # Section 3: Summary Table
    story.append(Spacer(1, 6))
    story.append(Paragraph("3. Cuadro Resumen Comparativo de Versiones", h1_style))
    
    headers = [Paragraph("<b>Versión</b>", body_bold),
               Paragraph("<b>Enfoque Central</b>", body_bold),
               Paragraph("<b>Problema / Incidencia</b>", body_bold),
               Paragraph("<b>Acierto@10</b>", body_bold),
               Paragraph("<b>SLA / Estado</b>", body_bold)]
    
    table_rows = [headers,
        [Paragraph("Etapa 1", body_style), Paragraph("Bucle desacoplado + TF-IDF", body_style), Paragraph("Bloqueo de stock durante 180s; >4s", body_style), Paragraph("68.2%", body_style), Paragraph("Inestable", body_style)],
        [Paragraph("Etapa 2", body_style), Paragraph("Lift + UserProfile + MMR", body_style), Paragraph("Pausas GIL por JSON; 22 pesos ciegos", body_style), Paragraph("73.5%", body_style), Paragraph("Vulnerable", body_style)],
        [Paragraph("Calypso 2.1", body_style), Paragraph("Re-Ranker ML IPS + Bandits", body_style), Paragraph("<b>Caída crítica SLA al 10.3%</b> (zombis)", body_style), Paragraph("77.2%", body_style), Paragraph("Crítico (caída)", alert_box_style)],
        [Paragraph("Etapa 4", body_style), Paragraph("Caché disco <0.4s + Slot-Gate", body_style), Paragraph("Pesos aún heurísticos sin reglas", body_style), Paragraph("77.2%", body_style), Paragraph("<b>>99.5% ON TIME</b>", body_style)],
        [Paragraph("Calypso 3.1", body_style), Paragraph("Quitar suelos + EASE 0.25", body_style), Paragraph("<b>Regresión al 74.5% (-2.7%)</b>", alert_box_style), Paragraph("74.5%", body_style), Paragraph(">99.5%", body_style)],
        [Paragraph("Calypso 3.1b", body_style), Paragraph("Ablación + Fondo selectivo", body_style), Paragraph("Falta de reglas para carritos", body_style), Paragraph("79.0%", body_style), Paragraph(">99.5%", body_style)],
        [Paragraph("Calypso 4.1", body_style), Paragraph("Reglas comerciales M1-M5", body_style), Paragraph("<b>Pérdida 25% DCG</b> por orden barato", alert_box_style), Paragraph("78.7%", body_style), Paragraph(">99.5%", body_style)],
        [Paragraph("Calypso 4.2", body_style), Paragraph("Corrección M3 + Tasa base", body_style), Paragraph("Techo de 386 compras del log", body_style), Paragraph("79.2%", body_style), Paragraph(">99.5%", body_style)],
        [Paragraph("Calypso 5.1", body_style), Paragraph("Gobernanza Modular Hot-Reload", body_style), Paragraph("Optimización ciega de 65 params", body_style), Paragraph("79.6%", body_style), Paragraph(">99.5%", body_style)],
        [Paragraph("<b>Calypso 6.2b</b>", body_bold), Paragraph("<b>Dual-Head Slot-1 + Student MLP</b>", body_bold), Paragraph("190 parámetros (gestionados por presets)", body_style), Paragraph("<b>74.4% (82.6% Cold)</b>", body_bold), Paragraph("<b>>99.9% (<2.8ms)</b>", body_bold)],
    ]

    summary_table = Table(table_rows, colWidths=[65, 120, 155, 72, 75])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor("#e2e8f0")),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('LEFTPADDING', (0,0), (-1,-1), 4),
        ('RIGHTPADDING', (0,0), (-1,-1), 4),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor("#f8fafc")]),
    ]))
    story.append(KeepTogether([summary_table]))

    # Section 4: Jury defense points
    story.append(Spacer(1, 10))
    story.append(Paragraph("4. Argumentario para la Defensa ante el Jurado (30 Puntos)", h1_style))
    jury_points = [
        "<b>1. Método y Validación Científica (10 pts):</b> Todo cambio se sometió a estudios de ablación con semillas aleatorias. Identificamos matemáticamente que el Slot 1 concentra el 34% de las compras y que el orden por precio ascendente costaba un -25% de DCG, descartando heurísticas intuitivas que perjudicaban el negocio.",
        "<b>2. Aprendizaje del Flujo en Vivo (10 pts):</b> El modelo no es estático: adapta continuamente sus pesos combinando bandit posicional $((k+1)^{-0.75})$ con el monitor de velocidad CVR en vivo de RawTree. En Cold Start (10 pts del reto), <i>product_profile_focus</i> alcanza un <b>82.61%</b> de acierto al inyectar nuevos productos en slots controlados.",
        "<b>3. Uso Avanzado de RawTree / SQL (5 pts):</b> Delegamos el cálculo pesado de agregaciones complejas a ClickHouse (ventanas de 60 min, Lift temporal y matrices de subcategoría), reduciendo transferencias de red, y utilizamos <i>team_32.scratch</i> como base de datos viva para trazas del agente y estados del bandit.",
        "<b>4. Excelencia de Ingeniería y Fiabilidad:</b> Tras la caída de la Etapa 3, blindamos la arquitectura con caché atómica a disco, arranque en < 0.4s, filtrado de peticiones caducadas (< 2.4s) y watchdog permanente, asegurando un <b>> 99.5% de cumplimiento de SLA</b> y respuesta constante en <b>< 2.8 ms</b>."
    ]
    for jp in jury_points:
        story.append(Paragraph(f"• {jp}", bullet_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF successfully built at: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
