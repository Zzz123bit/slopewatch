# -*- coding: utf-8 -*-
"""Generate the SlopeWatch status & roadmap document (DOCX)."""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
OUT = DOCS / "SlopeWatch_项目现状与计划.docx"

GREEN = RGBColor(0x17, 0x3B, 0x35)
INK = RGBColor(0x1F, 0x2B, 0x33)
MUTED = RGBColor(0x52, 0x67, 0x5D)
HEADER_FILL = "E8F1EC"
ZEBRA_FILL = "F7FAF8"

FONT = "Microsoft YaHei"


def _font(run, size=None, bold=None, color=None):
    run.font.name = FONT
    run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.bold = bold
    if color is not None:
        run.font.color.rgb = color


def _style(document, name, size, bold=False, color=INK, space_before=0, space_after=4):
    style = document.styles[name]
    style.font.name = FONT
    style.font.size = Pt(size)
    style.font.bold = bold
    style.font.color.rgb = color
    style.element.rPr.rFonts.set(qn("w:eastAsia"), FONT)
    style.paragraph_format.space_before = Pt(space_before)
    style.paragraph_format.space_after = Pt(space_after)


def body(document, text, *, size=10.5, bold=False, color=INK, space_after=4, align=None):
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(space_after)
    paragraph.paragraph_format.line_spacing = 1.35
    if align is not None:
        paragraph.alignment = align
    run = paragraph.add_run(text)
    _font(run, size=size, bold=bold, color=color)
    return paragraph


def bullet(document, text, *, size=10.5):
    paragraph = document.add_paragraph(style="List Bullet")
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(text)
    _font(run, size=size)
    return paragraph


def heading(document, text, level=1):
    paragraph = document.add_heading(level=level)
    run = paragraph.add_run(text)
    _font(run, size={1: 15, 2: 12, 3: 11}.get(level, 11), bold=True, color=GREEN)
    paragraph.paragraph_format.space_before = Pt(14 if level == 1 else 10)
    paragraph.paragraph_format.space_after = Pt(6)
    return paragraph


def shade(cell, color):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), color)
    tc_pr.append(shd)


def table(document, headers, rows, widths, *, font_size=9, zebra=True):
    tbl = document.add_table(rows=1, cols=len(headers))
    tbl.style = "Table Grid"
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    tbl.autofit = False
    header_row = tbl.rows[0]
    tr_pr = header_row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    tr_pr.append(repeat)
    header = header_row.cells
    for index, text in enumerate(headers):
        header[index].text = ""
        paragraph = header[index].paragraphs[0]
        run = paragraph.add_run(text)
        _font(run, size=font_size, bold=True, color=GREEN)
        shade(header[index], HEADER_FILL)
    for row_index, values in enumerate(rows):
        cells = tbl.add_row().cells
        for index, text in enumerate(values):
            cells[index].text = ""
            paragraph = cells[index].paragraphs[0]
            paragraph.paragraph_format.space_after = Pt(0)
            run = paragraph.add_run(str(text))
            _font(run, size=font_size)
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
    run = paragraph.add_run(text)
    _font(run, size=9, color=MUTED)
    return paragraph


def picture(document, path, caption, width=16.2):
    if not path.is_file():
        return
    document.add_picture(str(path), width=Cm(width))
    document.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    document.paragraphs[-1].paragraph_format.keep_with_next = True
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = paragraph.add_run(caption)
    _font(run, size=8.5, color=MUTED)
    paragraph.paragraph_format.space_after = Pt(8)


def add_page_number(document):
    footer = document.sections[0].footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = footer.add_run("坡镜智安 SlopeWatch · 项目现状与计划    第 ")
    _font(run, size=8, color=MUTED)
    field_begin = OxmlElement("w:fldChar"); field_begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText"); instruction.text = "PAGE"
    field_end = OxmlElement("w:fldChar"); field_end.set(qn("w:fldCharType"), "end")
    number_run = footer.add_run()
    _font(number_run, size=8, color=MUTED)
    number_run._r.append(field_begin)
    number_run._r.append(instruction)
    number_run._r.append(field_end)
    tail = footer.add_run(" 页")
    _font(tail, size=8, color=MUTED)


def build():
    document = Document()
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.2)
    section.right_margin = Cm(2.2)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(1.8)

    _style(document, "Normal", 10.5)
    _style(document, "Heading 1", 15, True, GREEN)
    _style(document, "Heading 2", 12, True, GREEN)
    _style(document, "List Bullet", 10.5)
    add_page_number(document)

    # ---------------- cover ----------------
    title = document.add_paragraph()
    title.paragraph_format.space_before = Pt(60)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run("坡镜智安 SlopeWatch")
    _font(run, size=26, bold=True, color=GREEN)

    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run("项目现状、端到端流程与后续计划")
    _font(run, size=15, color=MUTED)

    document.add_paragraph()
    meta = [
        ("代码仓库", "https://github.com/Zzz123bit/slopewatch"),
        ("当前提交", "64cbbad（main 分支，工作区干净）"),
        ("生成日期", "2026-11-28"),
        ("测试状态", "28 个自动化测试全部通过"),
        ("文档版本", "v1.0"),
    ]
    for key, value in meta:
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = paragraph.add_run(f"{key}：{value}")
        _font(run, size=10, color=INK)
        paragraph.paragraph_format.space_after = Pt(3)

    document.add_page_break()

    # ---------------- 1 概览 ----------------
    heading(document, "一、项目定位与当前结论", 1)
    body(document, "坡镜智安是一套面向室内小型边坡降雨试验的低成本视觉监测原型：用手机或普通相机拍摄"
                   "多视角照片，经 VGGT 重建两期三维坡面，自动配准后计算冲淤深度与体积，叠加语义类别，"
                   "生成可下载的巡检报告；同时提供网页、REST API 与带令牌的远程演示入口。")
    body(document, "当前结论：几何重建、配准、冲淤量化、语义分割、报告与网页链路均已跑通并有测试覆盖；"
                   "下一阶段的瓶颈不在代码，而在“真实试验数据的标注与验收”和“物理尺度自动标定”两件事。", bold=True)

    # ---------------- 2 架构 ----------------
    heading(document, "二、系统架构", 1)
    body(document, "系统按数据流分为六层，层与层之间通过文件或函数接口解耦，便于单独替换。")
    table(
        document,
        ["层次", "主要模块", "职责"],
        [
            ["采集与质控", "scripts/annotate.py、preannotate.py、slopewatch/uploads.py、quality.py",
             "照片打包上传、清晰度/曝光/分辨率检查、标注与预标注"],
            ["几何重建", "slopewatch/vggt_recon.py、reconstruction.py（COLMAP 备选）",
             "VGGT-1B 多视角推理、视角选择、权重校验、GPU/CPU 自动回退"],
            ["配准与基准", "slopewatch/photogram.py、registration.py、survey.py",
             "Sim(3) ICP、坡面平面定向、稳健去趋势；控制点/ArUco 解算（进阶）"],
            ["语义融合", "slopewatch/segnet.py、fusion.py、models/segmentation.onnx",
             "五类分割、掩膜几何对齐、多视投票投影、分类冲淤统计"],
            ["量化与出图", "slopewatch/core.py、visuals.py",
             "DoD、体积/深度/冲沟指标、三维坡面图、冲淤图、语义图谱"],
            ["服务与报告", "api.py、public_api.py、jobs.py、report.py、web/index.html、app.py",
             "任务队列、鉴权、文件下载、PDF 报告、网页与 Streamlit 前端"],
        ],
        [2.4, 6.6, 7.2],
    )
    note(document, "说明：survey.py/omega.py 是上一阶段留下的 VGGT-Ω 控制点方案，尚未接入照片任务，"
                   "作为“自动尺度标定”的现成基础保留。")
    screenshot = DOCS / "web_ui.png"
    cropped = DOCS / "web_ui_crop.png"
    if screenshot.is_file():
        with Image.open(screenshot) as source:
            source.crop((0, 0, source.width, min(source.height, 620))).save(cropped)
    picture(document, cropped, "图 1  远程页面：合成样例 / 上传点云 / 照片对比三种任务模式", width=12.6)

    # ---------------- 3 流程 ----------------
    heading(document, "三、端到端流程（照片对比）", 1)
    body(document, "一次完整任务分为七步，网页与 REST API 走的是同一套后端。")
    table(
        document,
        ["步骤", "做什么", "输入", "输出", "耗时参考"],
        [
            ["1 采集", "每期 8–32 张（建议 25–40 张）高重叠照片，固定路径，画面内放标尺或 ArUco 靶标",
             "手机/相机照片", "两个 ZIP", "10–20 分钟"],
            ["2 上传", "网页“照片对比”上传两个 ZIP，或调用 POST /api/v1/jobs/photo-compare",
             "before.zip / after.zip + 参数", "job_id", "数秒"],
            ["3 重建", "VGGT-1B 分别重建两期点云；显存不足自动回退 CPU",
             "照片", "两期点图与置信度", "GPU 约 20 秒/期；CPU 1–2 分钟/期"],
            ["4 配准", "Sim(3) ICP 求相似变换，坡面平面转 Z-up，再用 Huber 平面去趋势",
             "两期点云", "统一坐标系点云 + 残差", "数秒"],
            ["5 语义", "UNet 五类分割，按 VGGT 预处理几何对齐后多视投票投影到 DEM 网格",
             "雨后各视角照片", "类别栅格 + 分类统计", "数秒"],
            ["6 量化", "计算 DoD、冲淤体积、最大深度、冲沟候选与分区指标",
             "统一点云 + 类别", "metrics.json / change.csv", "数秒"],
            ["7 出报告", "渲染三维坡面、冲淤图、语义图谱并生成 PDF",
             "指标", "surface.png / dod.png / atlas.png / report.pdf", "数秒"],
        ],
        [1.4, 6.8, 3.6, 3.6, 2.8],
    )
    body(document, "实现上，3–7 步由 run_photo_job 在后台线程串行执行，任务状态与进度写入 "
                   "data/jobs/<job_id>/job.json，前端每 1.5 秒轮询一次。", size=9.5, color=MUTED)

    # ---------------- 4 完成度 ----------------
    heading(document, "四、完成度总览", 1)
    table(
        document,
        ["模块", "状态", "证据 / 说明"],
        [
            ["VGGT 双时相重建", "已完成", "真实权重推理通过；health 返回 vggt.available=true"],
            ["相似变换配准", "已完成", "解析地形做随机相似变换，体积恢复误差约 −3%"],
            ["冲淤量化与指标", "已完成", "体积、深度、面积、冲沟候选、变化速率均有测试"],
            ["三维/冲淤/图谱出图", "已完成", "docs/surface.png、dod.png、atlas.png"],
            ["PDF 巡检报告", "已完成", "docs/example_report.pdf（中文字体回退已处理）"],
            ["五类语义分割", "已完成", "mIoU 0.72；任务 semantic.source=model，五类全覆盖"],
            ["标注与微调工具链", "已完成", "预标注 + 画笔 + LabelMe 转换 + 微调脚本，真实照片预标注跑通"],
            ["网页与 REST API", "已完成", "三种任务模式、令牌鉴权、文件下载、远程隧道脚本"],
            ["自动化测试", "已完成", "pytest 28 passed"],
            ["真实场景精度", "待验证", "目前只有合成数据与几何仿真的量化结果"],
            ["物理尺度自动标定", "部分完成", "可手工填 mm/单位；ArUco 控制点方案未接入照片任务"],
            ["语义真实域适配", "部分完成", "模型由合成数据训练，真实照片存在明显域差距"],
        ],
        [4.2, 2.2, 9.8],
    )

    # ---------------- 5 已完成明细 ----------------
    heading(document, "五、已完成工作的关键指标", 1)
    heading(document, "5.1 几何与量化链路", 2)
    bullet(document, "解析地形随机尺度/旋转/平移后送入配准与差分：已知冲刷 2.78 L，恢复 2.70 L，误差约 −3%，最大深度一致，说明配准与体积积分本身可靠。")
    bullet(document, "端到端压力测试（真实 VGGT + 合成照片）：真值 3.99 L，恢复 1.50 L（约 38%），主要损失来自 1 m 拍摄距离下 5–10 cm 细沟的分辨极限，而非算法错误。")
    bullet(document, "运行耗时：GPU 可用时两期 4–8 视角约 40 秒；GPU 被桌面占用时自动回退 CPU，约 3–5 分钟。")
    heading(document, "5.2 语义分割", 2)
    table(
        document,
        ["指标", "数值"],
        [
            ["验证集 mIoU", "0.72"],
            ["像素准确率", "0.96"],
            ["各类 IoU（其他/冲沟/坍塌/防护/对照）", "0.57 / 0.46 / 0.38 / 0.97 / 0.98"],
            ["模型体积", "ONNX 4.3 MB；PyTorch checkpoint 4.2 MB"],
            ["训练数据", "程序化生成的 240 张多视角渲染图 + 像素掩膜"],
            ["单张推理", "CPU 约 0.1–0.2 秒"],
        ],
        [7.0, 9.2],
    )
    picture(document, DOCS / "segmentation_demo.png", "图 2  语义分割示例：输入 / 真值 / 预测（合成验证集）", width=16.0)
    picture(document, DOCS / "atlas.png", "图 3  照片任务输出的冲淤与语义图谱（Semantic source: model）", width=13.5)
    note(document, "在下载的风景参考照片上，模型会把大面积区域误判为冲沟/坍塌——这正是域差距的直接证据，"
                   "也是下一阶段必须用真实试验照片微调的原因。")

    # ---------------- 6 限制 ----------------
    heading(document, "六、已知限制与半完成项", 1)
    table(
        document,
        ["问题", "原因", "影响", "应对"],
        [
            ["尺度依赖人工", "VGGT 输出为任意尺度，任务只接受 mm/单位系数", "体积与深度可能整体偏大或偏小（体积随尺度三次方放大）",
             "接入 survey.py 的 ArUco 控制点解算，或在页面增加“标尺长度→尺度”工具"],
            ["语义域差距", "分割模型由合成渲染训练", "真实照片误判明显，分类统计不可直接用于结论", "用本机试验照片标注 100–200 张微调"],
            ["细沟分辨不足", "1 m 拍摄距离 + VGGT 稠密深度平滑", "5–10 cm 细沟深度被低估", "拉近拍摄、提高重叠、增加纹理靶标"],
            ["配准假设较强", "全局相似变换，默认两期相机路径相近", "视角差异大时配准失败或吸收真实变化", "固定机位、增加靶标；失败时检查 converged/残差"],
            ["仅支持两期", "任务模型为 before/after", "无法做多期时序与趋势", "扩展为多期任务与时间序列指标"],
            ["显存约束", "8 GB 笔记本 GPU 常被桌面占用", "8 视角易触发回退或 OOM", "默认 8 视角；必要时降为 6 或使用云 GPU"],
            ["演示安全", "共享团队令牌、本机 HTTP、无速率限制", "公网演示存在滥用风险", "隧道 + 强随机令牌；正式部署加限流与审计"],
        ],
        [3.4, 4.6, 4.6, 5.6],
    )

    # ---------------- 7 TODO ----------------
    heading(document, "七、待补清单（按优先级）", 1)
    body(document, "P0 是拿成绩必须完成的，P1 决定项目的可信度，P2 是加分项。", size=10)
    table(
        document,
        ["优先级", "任务", "产出 / 验收", "预估工时", "依赖"],
        [
            ["P0", "采集并标注 100–200 张真实试验照片", "data/real_masks；四类各 ≥15 张", "标注 2–3 小时（预标注后）", "相机/手机照片"],
            ["P0", "微调分割模型并验收", "val mIoU ≥ 0.75，冲沟/坍塌 IoU ≥ 0.5", "0.5 天", "上一条"],
            ["P0", "物理尺度自动标定", "页面填标尺长度即可换算，或 ArUco 控制点解算", "2–3 天", "survey.py 基础"],
            ["P0", "现场端到端验收 2 组", "两组试验的冲淤量、报告、图谱可复现", "1 天", "上两条"],
            ["P1", "精度评估报告", "与人工测量/TLS 对照的误差表", "1–2 天", "真实数据"],
            ["P1", "多期支持", "支持 3 期以上与时序曲线", "2–3 天", "当前两期流程"],
            ["P1", "失败检测与回退", "低纹理/模糊时给出重拍提示或切换 COLMAP", "1–2 天", "质量检查模块"],
            ["P1", "CI 与版本管理", "pytest 自动跑；数据/模型版本号", "0.5 天", "仓库权限"],
            ["P2", "云端部署与多租户", "域名、HTTPS、限流、日志审计", "3–5 天", "云主机"],
            ["P2", "采集辅助 App", "引导重叠度/曝光/拍摄路径", "1–2 周", "移动端开发"],
            ["P2", "报告模板定制", "按校赛/水保规范排版", "1 天", "模板需求"],
        ],
        [1.4, 4.4, 5.6, 3.2, 2.8],
    )

    # ---------------- 8 风险 ----------------
    heading(document, "八、主要风险与对策", 1)
    table(
        document,
        ["风险", "触发条件", "对策"],
        [
            ["真实场景分割不达标", "试验照片与合成数据差异大", "先标 30 张做小样本评估，再决定是否补样本或改类别体系"],
            ["尺度误差放大体积", "标尺/靶标未被正确识别", "多点标尺校验；报告中标注“未标定”并给出相对值"],
            ["配准失败或吸收变化", "两期视角差异大、纹理弱", "固定机位与拍摄路径；增加靶标；失败时提示人工复核"],
            ["推理资源不足", "8 GB 显存被占用", "自动回退 CPU；或改用云 GPU 运行重建"],
            ["依赖冲突", "onnx/protobuf 与 Streamlit 环境冲突", "训练在独立虚拟环境，运行时使用 .pt 回退"],
            ["时间不足", "临近比赛", "保 P0（真实数据、尺度、验收），砍 P2"],
        ],
        [4.2, 4.8, 8.0],
    )

    # ---------------- 9 运行手册 ----------------
    heading(document, "九、运行手册", 1)
    body(document, "以下命令均在项目根目录 D:\\tzzcup 执行。", size=9.5, color=MUTED)
    commands = [
        ("安装依赖", "python -m pip install -r requirements.txt"),
        ("VGGT 权重", "默认 D:\\models\\VGGT-1B\\model.safetensors，或设置 SLOPEWATCH_VGGT_CHECKPOINT"),
        ("启动 API", "uvicorn api:app --host 127.0.0.1 --port 8001"),
        ("启动桌面演示", "streamlit run app.py --server.address 127.0.0.1"),
        ("启动远程入口", "$env:SLOPEWATCH_API_TOKEN='<32 位以上令牌>'; uvicorn public_api:app --port 8011"),
        ("健康检查", "curl http://127.0.0.1:8001/api/v1/health"),
        ("预标注", "python scripts/preannotate.py --images data\\real_photos --masks data\\real_masks"),
        ("人工修正", "python scripts/annotate.py --images data\\real_photos --masks data\\real_masks"),
        ("微调模型", "python scripts/finetune_segmentation.py --images data\\real_photos --masks data\\real_masks --epochs 20"),
        ("运行测试", "python -m pytest -q"),
    ]
    for name, command in commands:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(2)
        run = paragraph.add_run(f"{name}：")
        _font(run, size=9.5, bold=True)
        run = paragraph.add_run(command)
        _font(run, size=9.5)
        run.font.name = "Consolas"
        run._element.rPr.rFonts.set(qn("w:eastAsia"), FONT)

    body(document, "提交任务示例（带令牌）：", size=9.5, color=MUTED)
    example = ("curl -X POST http://127.0.0.1:8011/api/v1/jobs/photo-compare "
               "-H \"Authorization: Bearer $TOKEN\" "
               "-F \"before_zip=@before.zip\" -F \"after_zip=@after.zip\" "
               "-F \"scale_mm_per_unit=1.0\" -F \"max_views=8\"")
    paragraph = document.add_paragraph()
    run = paragraph.add_run(example)
    _font(run, size=8.5)
    run.font.name = "Consolas"

    # ---------------- 10 交付物 ----------------
    heading(document, "十、交付物清单", 1)
    table(
        document,
        ["类别", "内容", "位置"],
        [
            ["代码", "重建、配准、量化、语义、接口、前端、脚本", "GitHub 仓库 main 分支（64cbbad）"],
            ["模型", "五类分割 ONNX 与 PyTorch checkpoint", "models/segmentation.onnx、models/segmentation.pt"],
            ["文档", "README、标注指南、本文件", "README.md、docs/annotation_guide.md、docs/SlopeWatch_项目现状与计划.docx"],
            ["示例", "界面截图、三维坡面、冲淤图、语义图谱、样例报告", "docs/*.png、docs/example_report.pdf"],
            ["数据", "合成演示数据、参考照片与标注", "data/（本地，未入库）"],
        ],
        [2.2, 7.0, 7.6],
    )

    # ---------------- 11 里程碑 ----------------
    heading(document, "十一、建议的两周冲刺", 1)
    table(
        document,
        ["时间", "目标", "交付"],
        [
            ["第 1–2 天", "采集并预标注 100–200 张真实照片，抽查质量", "掩膜数据集 + 标注报告"],
            ["第 3–4 天", "微调分割模型，评估 mIoU 与关键类", "更新 models/segmentation.pt，指标记录"],
            ["第 5–7 天", "接入尺度标定（标尺长度或 ArUco）", "页面新增尺度输入/自动解算"],
            ["第 8–10 天", "两组试验端到端验收，整理精度对照", "验收记录 + 报告样例"],
            ["第 11–12 天", "演示彩排：隧道、手机访问、下载报告", "比赛演示脚本"],
            ["第 13–14 天", "缓冲：修 bug、补文档、准备答辩材料", "发布 tag v1.0"],
        ],
        [2.4, 7.2, 7.2],
    )

    end = document.add_paragraph()
    end.paragraph_format.space_before = Pt(18)
    run = end.add_run("本文档由 AI 助手依据仓库当前状态生成；指标与命令均可在仓库对应提交中复核。")
    _font(run, size=8.5, color=MUTED)

    document.save(str(OUT))
    print("saved", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    build()
