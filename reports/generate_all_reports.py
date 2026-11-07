# -*- coding: utf-8 -*-
"""
generate_all_reports.py
Parses structured academic Markdown and builds publication-quality PDFs:
1. Predicting_YouTube_Video_Views_MLOps_Continuation_Labs_6_to_13.pdf
2. Predicting_YouTube_Video_Views_Complete_Report_Labs_1_to_13.pdf
"""

import os
import sys
import re

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
REPORTS_DIR = SCRIPT_DIR
sys.path.insert(0, SCRIPT_DIR)

import report_content

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable, Preformatted
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
            self.draw_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_decorations(self, page_count):
        self.saveState()
        self.setFont('Times-Roman', 8.5)
        self.setFillColor(colors.HexColor('#475569'))

        # Running header on pages > 1
        if self._pageNumber > 1:
            self.drawString(54, 755, 'Predicting YouTube Video Views: Regression & Production MLOps System')
            self.drawRightString(558, 755, 'Koppuravuri Sri Hinduja (241FA18284)')
            self.setStrokeColor(colors.HexColor('#CBD5E1'))
            self.setLineWidth(0.6)
            self.line(54, 749, 558, 749)

        # Running footer on all pages
        page_str = f'Page {self._pageNumber} of {page_count}'
        self.drawRightString(558, 36, page_str)
        self.drawString(54, 36, 'Department of Computer Science & Engineering | YouTube Views Prediction')
        self.setStrokeColor(colors.HexColor('#CBD5E1'))
        self.setLineWidth(0.6)
        self.line(54, 46, 558, 46)
        self.restoreState()


def get_styles():
    base = getSampleStyleSheet()
    return {
        'title': ParagraphStyle(
            'DocTitle',
            parent=base['Heading1'],
            fontName='Times-Bold',
            fontSize=16,
            leading=20,
            alignment=1, # Center
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=4
        ),
        'subtitle': ParagraphStyle(
            'DocSubTitle',
            parent=base['Normal'],
            fontName='Times-Bold',
            fontSize=10.5,
            leading=13.5,
            alignment=1,
            textColor=colors.HexColor('#1E3A8A'),
            spaceAfter=6
        ),
        'meta': ParagraphStyle(
            'DocMeta',
            parent=base['Normal'],
            fontName='Times-Roman',
            fontSize=9,
            leading=12.5,
            alignment=1,
            textColor=colors.HexColor('#334155'),
            spaceAfter=10
        ),
        'abstract_title': ParagraphStyle(
            'AbstractTitle',
            parent=base['Normal'],
            fontName='Times-BoldItalic',
            fontSize=9.5,
            leading=12.5,
            textColor=colors.HexColor('#0F172A'),
            spaceAfter=2
        ),
        'abstract_body': ParagraphStyle(
            'AbstractBody',
            parent=base['Normal'],
            fontName='Times-Italic',
            fontSize=8.5,
            leading=12,
            alignment=4, # Justified
            spaceAfter=8
        ),
        'sec': ParagraphStyle(
            'SecHeading',
            parent=base['Heading2'],
            fontName='Times-Bold',
            fontSize=11,
            leading=14,
            textColor=colors.HexColor('#0F172A'),
            spaceBefore=10,
            spaceAfter=3,
            keepWithNext=True
        ),
        'subsec': ParagraphStyle(
            'SubSecHeading',
            parent=base['Heading3'],
            fontName='Times-Bold',
            fontSize=9.5,
            leading=12.5,
            textColor=colors.HexColor('#1E3A8A'),
            spaceBefore=6,
            spaceAfter=2.5,
            keepWithNext=True
        ),
        'body': ParagraphStyle(
            'Body',
            parent=base['Normal'],
            fontName='Times-Roman',
            fontSize=8.8,
            leading=12.2,
            alignment=4,
            spaceAfter=4.5
        ),
        'bullet': ParagraphStyle(
            'Bullet',
            parent=base['Normal'],
            fontName='Times-Roman',
            fontSize=8.5,
            leading=11.5,
            leftIndent=14,
            firstLineIndent=-7,
            spaceAfter=2
        ),
        'table_cell': ParagraphStyle(
            'TableCell',
            parent=base['Normal'],
            fontName='Times-Roman',
            fontSize=7.5,
            leading=9.5,
            alignment=0
        ),
        'table_cell_bold': ParagraphStyle(
            'TableCellBold',
            parent=base['Normal'],
            fontName='Times-Bold',
            fontSize=7.5,
            leading=9.5,
            textColor=colors.white,
            alignment=0
        ),
        'table_cell_center': ParagraphStyle(
            'TableCellCenter',
            parent=base['Normal'],
            fontName='Times-Roman',
            fontSize=7.5,
            leading=9.5,
            alignment=1
        ),
        'code': ParagraphStyle(
            'CodeBlock',
            parent=base['Normal'],
            fontName='Courier',
            fontSize=6.8,
            leading=8.8,
            textColor=colors.HexColor('#1E293B')
        ),
        'callout': ParagraphStyle(
            'Callout',
            parent=base['Normal'],
            fontName='Times-Italic',
            fontSize=8.5,
            leading=11.5,
            leftIndent=10,
            rightIndent=10,
            textColor=colors.HexColor('#1E3A8A'),
            spaceAfter=5
        )
    }


def clean_xml_text(text):
    """Safely escapes text for ReportLab Paragraph markup."""
    # Convert math / symbols
    text = text.replace("&", "&amp;")
    text = text.replace("<=", "&le;").replace(">=", "&ge;")
    text = text.replace(r"\le", "&le;").replace(r"\ge", "&ge;")
    text = text.replace(r"\Delta", "&Delta;").replace(r"\rightarrow", "&rarr;")
    text = text.replace(r"\hat{\tilde{y}}", "y&#771;<sub>pred</sub>")
    text = text.replace(r"\tilde{y}", "y&#771;")
    text = text.replace(r"\hat{y}", "y&#770;")
    text = text.replace(r"\bar{\tilde{y}}", "y&#771;<sub>mean</sub>")
    text = text.replace(r"\bar{y}", "y<sub>mean</sub>")
    text = text.replace(r"\text{views}", "views")
    text = text.replace(r"\text{log1p}", "log1p")
    text = text.replace(r"\text{expm1}", "expm1")
    text = text.replace(r"\text{train}", "train")
    text = text.replace(r"\text{test}", "test")
    text = text.replace(r"\text{Challenger}", "Challenger")
    text = text.replace(r"\text{Champion}", "Champion")
    text = text.replace(r"\sup_x", "sup<sub>x</sub>")
    text = text.replace(r"\sum", "&Sigma;")
    text = text.replace(r"\sqrt", "&radic;")
    text = text.replace(r"\frac", "")
    
    # Simple latex cleaning
    text = re.sub(r'R\^2', 'R<sup>2</sup>', text)
    text = re.sub(r'\$R\^2\$', 'R<sup>2</sup>', text)
    text = re.sub(r'\$(.*?)\$', r'\1', text) # strip bare math $
    
    # Bold & Italic markdown conversion
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    
    # Inline code
    text = re.sub(r'`(.*?)`', r'<font name="Courier" size="7.5" color="#0F172A"><b>\1</b></font>', text)
    
    return text


def markdown_to_flowables(md_text, styles):
    flowables = []
    lines = md_text.split('\n')
    i = 0
    total_lines = len(lines)
    
    in_code_block = False
    code_lines = []
    
    in_table = False
    table_lines = []
    
    while i < total_lines:
        line = lines[i].rstrip()
        
        # Code block fence
        if line.startswith('```'):
            if in_code_block:
                # end code block
                in_code_block = False
                code_text = '\n'.join(code_lines)
                code_lines = []
                
                # Render code box as a single-cell table with background
                escaped_code = code_text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                p_code = Preformatted(code_text, styles['code'])
                box = Table([[p_code]], colWidths=[504])
                box.setStyle(TableStyle([
                    ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#F8FAFC')),
                    ('BOX', (0,0), (-1,-1), 0.6, colors.HexColor('#CBD5E1')),
                    ('TOPPADDING', (0,0), (-1,-1), 5),
                    ('BOTTOMPADDING', (0,0), (-1,-1), 5),
                    ('LEFTPADDING', (0,0), (-1,-1), 7),
                    ('RIGHTPADDING', (0,0), (-1,-1), 7),
                ]))
                flowables.append(Spacer(1, 4))
                flowables.append(box)
                flowables.append(Spacer(1, 6))
            else:
                in_code_block = True
                code_lines = []
            i += 1
            continue
            
        if in_code_block:
            code_lines.append(line)
            i += 1
            continue
            
        # Table parsing
        if line.startswith('|') and line.endswith('|'):
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
            i += 1
            continue
        else:
            if in_table:
                # process table
                in_table = False
                if len(table_lines) >= 3:
                    raw_rows = []
                    for t_line in table_lines:
                        # strip outer pipes and split
                        content = t_line.strip()[1:-1]
                        cols = [c.strip() for c in content.split('|')]
                        raw_rows.append(cols)
                    
                    # Header row
                    headers = raw_rows[0]
                    # Row 1 is divider |:---|:---|
                    data_rows = raw_rows[2:]
                    
                    num_cols = len(headers)
                    avail_width = 504.0
                    col_w = [avail_width / num_cols] * num_cols
                    
                    # Format cells
                    formatted_table_data = []
                    # header
                    hdr_cells = [Paragraph(clean_xml_text(h), styles['table_cell_bold']) for h in headers]
                    formatted_table_data.append(hdr_cells)
                    
                    for r in data_rows:
                        row_cells = []
                        for cell_idx, c in enumerate(r):
                            cell_text = clean_xml_text(c)
                            row_cells.append(Paragraph(cell_text, styles['table_cell']))
                        # Pad row if fewer cols
                        while len(row_cells) < num_cols:
                            row_cells.append(Paragraph("", styles['table_cell']))
                        formatted_table_data.append(row_cells[:num_cols])
                        
                    t = Table(formatted_table_data, colWidths=col_w)
                    t.setStyle(TableStyle([
                        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F172A')),
                        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
                        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                        ('TOPPADDING', (0,0), (-1,-1), 3.5),
                        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
                        ('LEFTPADDING', (0,0), (-1,-1), 4),
                        ('RIGHTPADDING', (0,0), (-1,-1), 4),
                        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
                        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E1')),
                    ]))
                    flowables.append(Spacer(1, 4))
                    flowables.append(t)
                    flowables.append(Spacer(1, 6))
                table_lines = []

        # Empty line
        if not line.strip():
            i += 1
            continue
            
        # Horizontal rule
        if line.startswith('---'):
            flowables.append(Spacer(1, 3))
            flowables.append(HRFlowable(width="100%", thickness=0.6, color=colors.HexColor('#CBD5E1'), spaceBefore=2, spaceAfter=5))
            i += 1
            continue

        # Document Title (# )
        if line.startswith('# '):
            t_text = clean_xml_text(line[2:])
            flowables.append(Paragraph(t_text, styles['title']))
            i += 1
            continue

        # Subtitle (## )
        if line.startswith('## '):
            st_text = clean_xml_text(line[3:])
            flowables.append(Paragraph(st_text, styles['subtitle']))
            i += 1
            continue

        # Section Header (### )
        if line.startswith('### '):
            sec_text = clean_xml_text(line[4:])
            flowables.append(Spacer(1, 4))
            flowables.append(Paragraph(sec_text, styles['sec']))
            i += 1
            continue

        # Subsection Header (#### )
        if line.startswith('#### '):
            subsec_text = clean_xml_text(line[5:])
            flowables.append(Spacer(1, 2))
            flowables.append(Paragraph(subsec_text, styles['subsec']))
            i += 1
            continue

        # Bullet list
        if line.startswith('- ') or line.startswith('* '):
            b_text = clean_xml_text(line[2:])
            flowables.append(Paragraph(f'&bull; {b_text}', styles['bullet']))
            i += 1
            continue
            
        # Metadata lines (e.g. **Author:** ...)
        if line.startswith('**Author:**') or line.startswith('**Email:**') or line.startswith('**Institution:**') or line.startswith('**Date:**') or line.startswith('**Repository:**'):
            m_text = clean_xml_text(line)
            flowables.append(Paragraph(m_text, styles['meta']))
            i += 1
            continue

        # Normal paragraph
        p_text = clean_xml_text(line)
        flowables.append(Paragraph(p_text, styles['body']))
        i += 1

    return flowables


def build_pdf(output_path, markdown_content, doc_title):
    styles = get_styles()
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54,
        title=doc_title,
        author="Koppuravuri Sri Hinduja (241FA18284)"
    )
    
    flowables = markdown_to_flowables(markdown_content, styles)
    doc.build(flowables, canvasmaker=NumberedCanvas)
    print(f"Successfully built PDF: {output_path} ({os.path.getsize(output_path)} bytes)")


def main():
    print("=" * 70)
    print("Starting Comprehensive Academic Report Generation (Labs 1-13 & 6-13)")
    print("=" * 70)

    # 1. Output file paths
    cont_md_path = os.path.join(REPORTS_DIR, "Predicting_YouTube_Video_Views_MLOps_Continuation_Labs_6_to_13.md")
    comp_md_path = os.path.join(REPORTS_DIR, "Predicting_YouTube_Video_Views_Complete_Report_Labs_1_to_13.md")
    cont_pdf_path = os.path.join(REPORTS_DIR, "Predicting_YouTube_Video_Views_MLOps_Continuation_Labs_6_to_13.pdf")
    comp_pdf_path = os.path.join(REPORTS_DIR, "Predicting_YouTube_Video_Views_Complete_Report_Labs_1_to_13.pdf")

    # 2. Get Markdown content
    cont_md = report_content.get_continuation_markdown()
    comp_md = report_content.get_complete_markdown()

    # 3. Write Markdown files
    with open(cont_md_path, 'w', encoding='utf-8') as f:
        f.write(cont_md)
    print(f"Saved Markdown Continuation: {cont_md_path} ({os.path.getsize(cont_md_path)} bytes)")

    with open(comp_md_path, 'w', encoding='utf-8') as f:
        f.write(comp_md)
    print(f"Saved Markdown Complete: {comp_md_path} ({os.path.getsize(comp_md_path)} bytes)")

    # 4. Build PDF Continuation Report
    print("\nBuilding Continuation PDF Report (Labs 6 to 13)...")
    build_pdf(cont_pdf_path, cont_md, "Predicting YouTube Video Views - MLOps Continuation Report")

    # 5. Build PDF Complete Report
    print("\nBuilding Unified Complete PDF Report (Labs 1 to 13)...")
    build_pdf(comp_pdf_path, comp_md, "Predicting YouTube Video Views - Complete End-to-End Report")

    print("\n" + "=" * 70)
    print("ALL ACADEMIC REPORT DELIVERABLES GENERATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
