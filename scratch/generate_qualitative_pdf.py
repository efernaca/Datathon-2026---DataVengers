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
            self.drawRightString(541, 800, "Resumen Cualitativo de Calypso")
            self.setStrokeColor(colors.HexColor("#CCCCCC"))
            self.setLineWidth(0.5)
            self.line(54, 792, 541, 792)

        # Footer (all pages)
        page_str = f"Página {self._pageNumber} de {page_count}"
        self.drawRightString(541, 35, page_str)
        self.drawString(54, 35, "Confidencial · Presentación Cualitativa al Jurado")
        self.setStrokeColor(colors.HexColor("#CCCCCC"))
        self.setLineWidth(0.5)
        self.line(54, 47, 541, 47)
        self.restoreState()

def build_pdf():
    repo_root = Path(r"C:\Users\megia\Desktop\stdw\Datathon-2026---DataVengers")
    pdf_path = repo_root / "Presentacion" / "Resumen_Cualitativo_Calypso.pdf"
    
    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
    )
    
    primary_color = colors.HexColor("#184c78")
    dark_text = colors.HexColor("#1a1a1a")
    
    title_style = ParagraphStyle(
        "DocTitle",
        fontName="Times-Bold",
        fontSize=20,
        leading=24,
        textColor=primary_color,
        alignment=1,
        spaceAfter=5,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="Times-Italic",
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor("#333333"),
        alignment=1,
        spaceAfter=6,
    )
    meta_style = ParagraphStyle(
        "DocMeta",
        fontName="Times-Roman",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#555555"),
        alignment=1,
        spaceAfter=12,
    )
    h1_style = ParagraphStyle(
        "H1",
        fontName="Times-Bold",
        fontSize=12.5,
        leading=16.5,
        textColor=primary_color,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True,
    )
    body_style = ParagraphStyle(
        "Body",
        fontName="Times-Roman",
        fontSize=10,
        leading=14.5,
        textColor=dark_text,
        spaceAfter=6,
    )
    synopsis_style = ParagraphStyle(
        "Synopsis",
        fontName="Times-Italic",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#1e293b"),
    )
    takeaway_style = ParagraphStyle(
        "Takeaway",
        fontName="Times-Roman",
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor("#0f172a"),
    )
    bullet_style = ParagraphStyle(
        "Bullet",
        fontName="Times-Roman",
        fontSize=9.5,
        leading=14,
        textColor=dark_text,
        leftIndent=14,
        spaceAfter=4,
    )

    story = []

    # Title & Metadata
    story.append(Paragraph("Resumen Cualitativo: El Viaje de Calypso", title_style))
    story.append(Paragraph("De la Intuición a la Excelencia en Ingeniería de Sistemas y Aprendizaje Automático", subtitle_style))
    story.append(Paragraph("<b>Equipo DataVengers (Team 32)</b> · Datathon FME 2026 (UPC Barcelona) · RawTree Challenge<br/>15 de noviembre de 2026", meta_style))
    story.append(HRFlowable(width="100%", thickness=1, color=primary_color, spaceAfter=10))

    # Narrative Synopsis Box
    synopsis_text = (
        "<b>Sinopsis Narrativa:</b> El desarrollo del recomendador Calypso por parte de DataVengers no ha sido una mera "
        "acumulación de ecuaciones, sino un viaje de madurez técnica y científica. A lo largo de diez iteraciones, "
        "el equipo aprendió a sustituir la intuición por el rigor experimental, transformó una caída crítica de disponibilidad "
        "en una infraestructura invulnerable y diseñó una solución definitiva capaz de entender tanto la intención inmediata "
        "del comprador como las oportunidades invisibles del catálogo frío."
    )
    synopsis_table = Table([[Paragraph(synopsis_text, synopsis_style)]], colWidths=[487])
    synopsis_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f8fafc")),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
        ('LINELEFT', (0,0), (-1,-1), 3.5, primary_color),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(synopsis_table)
    story.append(Spacer(1, 10))

    # Section 1
    story.append(Paragraph("1. De la Intuición a la Realidad Operativa (Etapas 1 y 2)", h1_style))
    p1 = (
        "En los primeros compases del reto, el planteamiento fue el intuitivo en cualquier sistema de comercio electrónico: "
        "conectar señales lógicas y evidentes (quien consulta este artículo suele adquirir aquel, rescate de carritos abandonados "
        "y segmentación por historial de navegación y compra).<br/><br/>"
        "Sin embargo, el entorno de producción en tiempo real nos dio la primera llamada de atención: <b>un algoritmo conceptualmente "
        "brillante sobre el papel no sirve de nada si la infraestructura se ahoga en el mundo real</b>. Intentar descargar y analizar "
        "perfiles de cientos de miles de usuarios en vivo provocaba bloqueos del intérprete de Python (<i>GIL freeze</i>) y amenazaba "
        "el límite inflexible de los 3.000 ms."
    )
    story.append(Paragraph(p1, body_style))

    box1 = Table([[Paragraph("<b>Lección aprendida:</b> La inferencia en tiempo real debe ser pura, inmediata y vivir exclusivamente en memoria; la analítica pesada y las agregaciones masivas deben ocurrir en segundo plano y lejos del cliente.", takeaway_style)]], colWidths=[487])
    box1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor("#f1f5f9")),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('LINELEFT', (0,0), (-1,-1), 2.5, colors.HexColor("#475569")),
        ('BOX', (0,0), (-1,-1), 0.5, colors.HexColor("#cbd5e1")),
    ]))
    story.append(box1)
    story.append(Spacer(1, 6))

    # Section 2
    story.append(Paragraph("2. La Crisis como Motor de Resiliencia (Etapa 3 a Etapa 4)", h1_style))
    p2 = (
        "El punto de inflexión del proyecto se produjo en Calypso 2.1. Confiados en haber incorporado Machine Learning de vanguardia "
        "con corrección de sesgo posicional (IPS) y algoritmos bandit, un reinicio imprevisto de la máquina destapó una vulnerabilidad crítica: "
        "el servicio intentaba procesar cientos de peticiones antiguas que ya habían caducado, provocando un efecto bola de nieve que hundió "
        "el cumplimiento del SLA al <b>10.3% de ON TIME</b>.<br/><br/>"
        "En lugar de aplicar un parche superficial, el equipo redefinió por completo la roca estructural del proyecto en la Etapa 4:"
    )
    story.append(Paragraph(p2, body_style))
    story.append(Paragraph("• <b>Memoria atómica en disco:</b> Dotamos al sistema de persistencia inmediata (<i>snapshot_v4_latest.pkl</i>), permitiendo recuperarse y responder al 100% del tráfico en <b>menos de 0.4 segundos</b>.", bullet_style))
    story.append(Paragraph("• <b>Inmunidad contra peticiones zombi:</b> Filtramos estrictamente las solicitudes con más de 2.400 ms de antigüedad, garantizando que ni un solo ciclo de CPU se malgaste en tráfico caducado.", bullet_style))
    story.append(Paragraph("• <b>Arquitectura Slot-Gated:</b> Blindamos la certeza de los slots 0 al 6 para precisión contrastada, relegando la experimentación UCB a la cola (slots 7 al 9).", bullet_style))
    story.append(Paragraph("Aquel tropiezo operativo se convirtió en nuestra mayor ventaja competitiva: <b>una fiabilidad superior al 99.5% a prueba de cualquier contingencia</b>.", body_style))

    # Section 3
    story.append(Paragraph("3. El Método Científico contra la Falsa Intuición (Calypso 3.1 a 4.2)", h1_style))
    p3 = "En esta fase, DataVengers desterró las conjeturas e instauró una disciplina férrea basada en <b>estudios controlados de ablación</b>:"
    story.append(Paragraph(p3, body_style))
    story.append(Paragraph("• <b>El tropiezo de Calypso 3.1:</b> Intentar modificar cuatro variables estructurales a la vez hizo caer el rendimiento del 77.2% al 74.5%. La ablación sistemática de la versión 3.1b nos enseñó a aislar cada cambio hasta descubrir una máxima fundamental: <i>«Nunca recomiendes un producto que no tenga un motivo contrastado para estar ahí»</i> (penalizar los productos de relleno irrelevantes disparó la efectividad al 79.0%).", bullet_style))
    story.append(Paragraph("• <b>El desengaño del precio en Calypso 4.1:</b> La intuición nos decía que ordenar los productos recomendados de más barato a más caro atraería más compras. Los datos experimentales nos dieron un golpe demoledor: la ordenación por precio destruía un <b>25% del DCG de compra</b>. ¿Por qué? Porque el primer puesto del carrusel concentra por sí solo el <b>34% de todas las compras</b>, y colocar el artículo más barato apartaba de ese escaparate de oro al producto que el modelo sabía que el usuario realmente deseaba. El precio nunca debe dictar la relevancia; solo actúa como desempate entre iguales.", bullet_style))

    # Section 4
    story.append(Paragraph("4. Madurez Industrial y Aprendizaje Profundo (Calypso 5.1 a 6.2b)", h1_style))
    p4 = "La etapa de consolidación transformó a Calypso en un motor industrial de alto nivel:"
    story.append(Paragraph(p4, body_style))
    story.append(Paragraph("• <b>Gobernanza en Caliente (5.1):</b> Desacoplamos la configuración del código mediante perfiles modulares con recarga en memoria (<i>hot-reload</i>), permitiendo reorientar la política comercial del sistema sin interrumpir el servicio ni perder peticiones.", bullet_style))
    story.append(Paragraph("• <b>Obsesión por el Primer Impacto (6.2b):</b> Diseñamos una arquitectura <i>Dual-Head Slot-1 Precision</i>, dedicando una cabecera especializada con 64 variables y mitigación de riesgo secuencial para asegurar la excelencia en el primer slot.", bullet_style))
    story.append(Paragraph("• <b>Destilación Neuronal y Cold Start Exponencial:</b> Entrenamos un modelo maestro complejo sobre miles de compras históricas y lo destilamos en una red neuronal compacta y ultrarrápida (Student MLP en NumPy puro). El resultado fue contundente: un <b>74.41% de acierto global</b> y un colosal <b>82.61% de éxito en Cold Start</b>, preparando al sistema para dominar el flujo ciego de la ronda final.", bullet_style))

    # Section 5
    story.append(Paragraph("5. Conclusión Estratégica", h1_style))
    p5 = (
        "El valor de <b>Calypso 6.2b</b> no radica en haber acertado a la primera, sino en haber demostrado capacidad para "
        "<b>fallar rápido, medir con rigor científico, derribar sesgos intuitivos y construir una ingeniería inquebrantable</b>.<br/><br/>"
        "Llegamos a la final con un recomendador que responde en menos de 2.8 milisegundos, que jamás colapsa ante imprevistos de red "
        "y que conecta con precisión matemática lo que el comprador busca con la mejor oferta del catálogo."
    )
    story.append(Paragraph(p5, body_style))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Qualitative PDF successfully generated at: {pdf_path}")

if __name__ == "__main__":
    build_pdf()
