# -*- coding: utf-8 -*-
"""Shared python-docx styling helpers for the SlopeWatch reports."""

from __future__ import annotations

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

GREEN = RGBColor(0x17, 0x3B, 0x35)
INK = RGBColor(0x1F, 0x2B, 0x33)
MUTED = RGBColor(0x52, 0x67, 0x5D)
HEADER_FILL = "E8F1EC"
ZEBRA_FILL = "F7FAF8"
FONT = "Microsoft YaHei"


def style_run(run, size=None, bold=None, color=None, mono=False):
    run.font.name = "Consolas" if mono else FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def new_document(footer_label="坡镜智安 SlopeWatch · 技术选型与框架清单"):
    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(1.8)
    for name, size, bold in (("Normal", 10.5, False), ("Heading 1", 15, True),
                             ("Heading 2", 12, True), ("List Bullet", 10.5, False)):
        style = document.styles[name]
        style.font.name = FONT
        style.font.size = Pt(size)
        style.font.bold = bold
        style.font.color.rgb = GREEN if bold else INK
        style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
        style.paragraph_format.space_before = Pt(0)
        style.paragraph_format.space_after = Pt(4)
    add_page_number(document, footer_label)
    return document


def add_page_number(document, label):
    footer = document.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(footer.add_run(label + "    第 "), size=8, color=MUTED)
    begin = OxmlElement("w:fldChar"); begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText"); instruction.text = "PAGE"
    end = OxmlElement("w:fldChar"); end.set(qn("w:fldCharType"), "end")
    run = footer.add_run()
    style_run(run, size=8, color=MUTED)
    run._r.append(begin); run._r.append(instruction); run._r.append(end)
    style_run(footer.add_run(" 页"), size=8, color=MUTED)


def body(document, text, *, size=10.5, bold=False, color=INK, space_after=4, align=None):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.line_spacing = 1.35
    if align is not None:
        paragraph.alignment = align
    style_run(paragraph.add_run(text), size=size, bold=bold, color=color)
    return paragraph


def bullet(document, text, *, size=10.5):
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(2)
    style_run(paragraph.add_run(text), size=size)
    return paragraph


def heading(document, text, level=1):
    paragraph = document.add_heading(level=level)
    style_run(paragraph.add_run(text), size={1: 15, 2: 12, 3: 11}.get(level, 11),
              bold=True, color=GREEN)
    paragraph.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    paragraph.paragraph_format.space_after = Pt(6)
    return paragraph


def code_block(document, text, size=8.5):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(6)
    paragraph.paragraph_format.left_indent = Cm(0.3)
    style_run(paragraph.add_run(text), size=size, mono=True)
    return paragraph


def shade(cell, color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tc_pr.append(shd)


def table(document, headers, rows, widths, *, font_size=8.8, zebra=True):
    tbl = document.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    header_row = tbl.rows[0]
    tr_pr = header_row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader"); repeat.set(qn("w:val"), "true"); tr_pr.append(repeat)
    for index, text in enumerate(headers):
        cell = header_row.cells[index]
        cell.text = ""
        style_run(cell.paragraphs[0].add_run(text), size=font_size, bold=True, color=GREEN)
        shade(cell, HEADER_FILL)
    for row_index, values in enumerate(rows):
        cells = tbl.add_row().cells
        for index, text in enumerate(values):
            cells[index].text = ""
            paragraph = cells[index].paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            style_run(paragraph.add_run(str(text)), size=font_size)
            if zebra and row_index % 2 == 1:
                shade(cells[index], ZEBRA_FILL)
    for row in tbl.rows:
        row_pr = row._tr.get_or_add_trPr()
        row_pr.append(OxmlElement("w:cantSplit"))
        for index, width in enumerate(widths):
            row.cells[index].width = Cm(width)
    return tbl


def note(document, text):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(8)
    style_run(paragraph.add_run(text), size=8.8, color=MUTED)
    return paragraph


def picture(document, path, caption, width=15.5):
    path = str(path)
    import os

    if not os.path.isfile(path):
        return
    document.add_picture(path, width=Cm(width))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.paragraphs[-1].paragraph_format.keep_with_next = True
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(paragraph.add_run(caption), size=8.5, color=MUTED)
    paragraph.paragraph_format.space_after = Pt(8)
