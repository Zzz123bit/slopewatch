from __future__ import annotations

import io
import json
from html import escape
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _register_cjk_font() -> str:
    windows_font = Path("C:/Windows/Fonts/msyh.ttc")
    if windows_font.is_file():
        font_name = "SlopeWatchMicrosoftYaHei"
        if font_name not in pdfmetrics.getRegisteredFontNames():
            pdfmetrics.registerFont(TTFont(font_name, str(windows_font), subfontIndex=0))
        return font_name
    font_name = "STSong-Light"
    if font_name not in pdfmetrics.getRegisteredFontNames():
        pdfmetrics.registerFont(UnicodeCIDFont(font_name))
    return font_name


def _xml_text(value) -> str:
    return escape(str(value)).replace("_", "&#95;")


def build_report_pdf(metrics: dict, gullies: list[dict], metadata: dict | None = None) -> bytes:
    font_name = _register_cjk_font()
    output = io.BytesIO()
    document = SimpleDocTemplate(
        output,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=17 * mm,
        bottomMargin=17 * mm,
        title="坡镜智安边坡巡检摘要",
        author="坡镜智安",
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "SlopeWatchTitle",
        parent=styles["Title"],
        fontName=font_name,
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#173B35"),
    )
    heading_style = ParagraphStyle(
        "SlopeWatchHeading",
        parent=styles["Heading2"],
        fontName=font_name,
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#173B35"),
        spaceBefore=8,
        spaceAfter=5,
    )
    body_style = ParagraphStyle(
        "SlopeWatchBody",
        parent=styles["BodyText"],
        fontName=font_name,
        fontSize=9,
        leading=14,
    )
    story = [Paragraph("边坡时相形貌分析摘要", title_style), Spacer(1, 5 * mm)]

    metadata = metadata or {}
    story.append(Paragraph("试验记录", heading_style))
    if metadata:
        metadata_rows = [["字段", "内容"]]
        for key, value in metadata.items():
            if isinstance(value, (dict, list)):
                value = json.dumps(value, ensure_ascii=False)
            metadata_rows.append([Paragraph(_xml_text(key), body_style), Paragraph(_xml_text(value), body_style)])
        metadata_table = Table(metadata_rows, colWidths=[45 * mm, 119 * mm], repeatRows=1)
        metadata_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2EF")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD6D2")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        story.append(metadata_table)
    else:
        story.append(Paragraph("无试验参数", body_style))
    story.extend([Spacer(1, 3 * mm), Paragraph("主要指标", heading_style)])

    metric_rows = [["指标", "数值", "单位或说明"]]
    definitions = [
        ("冲刷体积（阈值以上）", "erosion_volume_l", "L"),
        ("堆积体积（阈值以上）", "deposition_volume_l", "L"),
        ("未滤波冲刷体积", "erosion_volume_raw_l", "L"),
        ("未滤波堆积体积", "deposition_volume_raw_l", "L"),
        ("最大冲刷深度", "max_erosion_depth_mm", "mm"),
        ("平均冲刷深度（阈值以上）", "mean_erosion_depth_mm", "mm"),
        ("最大堆积高度", "max_deposition_height_mm", "mm"),
        ("变化区域面积", "changed_area_m2", "m^2"),
        ("冲沟候选数量", "gully_count_est", "个"),
        ("最大变化速率", "max_change_rate_mm_per_hour", "mm/h"),
        ("变化阈值", "erosion_threshold_mm", "mm"),
        ("网格间距", "grid_step_mm", "mm"),
    ]
    for label, key, unit in definitions:
        value = metrics.get(key)
        if value is None:
            continue
        metric_rows.append([label, str(value), unit])
    table = Table(metric_rows, colWidths=[66 * mm, 38 * mm, 60 * mm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("FONTNAME", (0, 0), (-1, -1), font_name),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2EF")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#173B35")),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD6D2")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 7),
                ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)

    story.append(Paragraph("冲刷区候选", heading_style))
    if gullies:
        gully_rows = [["编号", "面积 (mm^2)", "长度估计 (mm)", "宽度估计 (mm)", "最大深度 (mm)"]]
        for gully in gullies[:20]:
            gully_rows.append(
                [
                    str(gully["id"]),
                    str(gully["area_mm2"]),
                    str(gully["length_mm_est"]),
                    str(gully["width_mm_est"]),
                    str(gully["max_depth_mm"]),
                ]
            )
        gully_table = Table(gully_rows, colWidths=[15 * mm, 32 * mm, 36 * mm, 36 * mm, 40 * mm], repeatRows=1)
        gully_table.setStyle(
            TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), font_name),
                    ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#EAF2EF")),
                    ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#CAD6D2")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 5),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(gully_table)
    else:
        story.append(Paragraph("当前阈值下未提取到冲刷区候选。", body_style))

    story.extend(
        [
            Spacer(1, 5 * mm),
            Paragraph(
                "说明：冲沟长度和宽度由变化区域的平面形状估算。结果适用于共同坐标系下的单层坡面高程点云。请结合标尺、人工测量和试验记录核验。",
                body_style,
            ),
        ]
    )
    document.build(story)
    return output.getvalue()

