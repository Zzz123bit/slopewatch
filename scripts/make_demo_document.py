# -*- coding: utf-8 -*-
"""Generate the demo execution handbook (DOCX)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt

from docx_style import (GREEN, MUTED, body, bullet, code_block, heading, new_document,
                        note, style_run, table)

DOCS = Path(__file__).resolve().parents[1] / "docs"
OUT = DOCS / "SlopeWatch_演示手册.docx"


def build():
    document = new_document("坡镜智安 SlopeWatch · 演示手册")

    title = document.add_paragraph()
    title.paragraph_format.space_before = Pt(52)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(title.add_run("坡镜智安 SlopeWatch"), size=26, bold=True, color=GREEN)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(subtitle.add_run("演示设计与执行手册"), size=15, color=MUTED)

    document.add_paragraph()
    for line in [
        "5 分钟标准版 · 60 秒电梯版 · 故障兜底",
        "配套脚本：scripts/run_demo.py",
        "版本 v1.0 · 2026-11-28",
    ]:
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style_run(paragraph.add_run(line), size=10)
        paragraph.paragraph_format.space_after = Pt(3)
    document.add_page_break()

    heading(document, "一、演示定位", 1)
    body(document, "一句话：拍两组照片，几分钟后拿到“这场降雨冲走多少土、冲在哪里、属于哪类地貌”的报告。")
    body(document, "全程反复回扣三个记忆点：", bold=True)
    bullet(document, "便宜：手机或普通相机就能拍，不需要全站仪和专业测绘。")
    bullet(document, "快：上传到出报告几分钟，现场可完整演示。")
    bullet(document, "能解释：不只给体积数字，还给三维坡面、冲淤图与语义图谱。")

    heading(document, "二、5 分钟标准版分镜", 1)
    table(
        document,
        ["时间", "画面 / 操作", "讲稿要点", "预期结果"],
        [
            ["0:00–0:30", "网页首页", "边坡巡检的痛点是“想经常看，但请不起测绘队”。我们让普通照片变成可量化的巡检报告。", "建立预期"],
            ["0:30–1:00", "两张照片 / 拍摄示意", "雨前雨后各拍一组：固定路线、重叠 70%、画面放一把标尺。", "说明门槛低"],
            ["1:00–1:30", "网页「照片对比」选两个 ZIP", "把两组照片传上去。", "出现任务 ID 与进度条"],
            ["1:30–2:30", "进度条 + 架构图", "后台先做 VGGT 三维重建，不需要标定相机；再自动配准两期；然后算高程差。", "进度到 100%"],
            ["2:30–3:30", "三维坡面 → 冲淤图", "红色淤积、蓝色冲刷；这次冲刷约 0.35 L、最深 1.6 cm、20 处冲沟候选（以现场读数为准）。", "指标与图"],
            ["3:30–4:10", "语义图谱 + 下载 PDF", "不仅知道变了多少，还知道变在哪类地貌：冲沟、坍塌、防护区分别统计。", "atlas + PDF"],
            ["4:10–4:40", "手机扫码", "评委可以扫码，用自己的手机提交一次。", "远程页面打开"],
            ["4:40–5:00", "局限与下一步", "尺度目前手工标定，下一步接 ArUco 自动解算；语义模型正在用真实照片微调。", "诚实收尾"],
        ],
        [2.0, 4.2, 7.6, 2.6],
        font_size=8.4,
    )

    heading(document, "三、60 秒电梯版", 1)
    table(
        document,
        ["时间", "内容"],
        [
            ["0–10 秒", "痛点：边坡想高频巡检，但测绘贵、慢。"],
            ["10–25 秒", "亮冲淤图：拍两组照片，几分钟得到这张图与体积。"],
            ["25–45 秒", "怎么做到：VGGT 免标定重建 + 自动配准 + 地貌规则语义。"],
            ["45–60 秒", "仓库二维码 + 一句局限（尺度需标尺标定）。"],
        ],
        [2.2, 14.2],
        font_size=8.6,
    )

    heading(document, "四、现场准备清单", 1)
    table(
        document,
        ["类别", "项目", "检查方式"],
        [
            ["软件", "依赖、权重、分割模型", "运行 pytest（31 passed）；health 里 vggt/segmentation 均 ready"],
            ["数据", "before.zip / after.zip（各 8–12 张）", "提前放桌面，先自测一次"],
            ["兜底", "预生成结果、60 秒录屏", "docs/demo/quick、docs/demo/photo"],
            ["远程", "隧道地址 + 令牌二维码", "手机流量网络实测一次"],
            ["物料", "打印 PDF、A3 冲淤图、二维码", "各 2 份"],
            ["设备", "笔记本、手机、充电器、扩展坞", "电量 >80%，关闭系统更新"],
        ],
        [2.2, 7.0, 7.2],
        font_size=8.6,
    )

    heading(document, "五、故障兜底", 1)
    table(
        document,
        ["故障", "现象", "处理"],
        [
            ["GPU 被占", "重建进度很久不动", "系统自动回退 CPU（3–5 分钟）；或直接切预生成结果"],
            ["网络/隧道失效", "手机打不开页面", "本机网页 + 录屏；备用热点"],
            ["上传超时", "页面报连接中断", "减少到 8 张；或用「合成样例」3 秒出结果"],
            ["权重/依赖异常", "health 显示不可用", "播放录屏 + 展示 docs/demo/photo 的 PDF 与图"],
            ["投影故障", "黑屏", "打印的 PDF 与 A3 图顶上"],
        ],
        [3.2, 6.2, 7.0],
        font_size=8.6,
    )
    body(document, "兜底原则：先讲结论，再修系统；永远不要让评委盯着报错。", bold=True)

    heading(document, "六、必答题", 1)
    table(
        document,
        ["问题", "回答"],
        [
            ["精度多少？", "几何链路在解析地形上体积误差约 −3%；端到端受重建分辨率限制，1 m 距离下 5–10 cm 细沟会低估，真实数据按标尺复核。"],
            ["和全站仪/TLS 比？", "不比绝对精度，比频次与覆盖：TLS 一次几千元几小时，我们一部手机几分钟，适合高频普查，异常再上高精度手段复核。"],
            ["尺度怎么定？", "画面放已知长度标尺或 ArUco；当前手工填 mm/单位，下一步自动解算（survey.py 已有控制点基础）。"],
            ["语义标签哪来？", "几何规则自监督（冲沟/坍塌）+ 公开 CC-BY 数据预训练 + 人工修正 100–200 张（补防护/对照）。"],
            ["为什么用 VGGT？", "免相机标定、免 SfM 初始化，普通手机照片直接前馈出位姿与点图；代价是尺度任意，需要外部标定。"],
            ["能商用吗？", "链路可以；VGGT 需换 Commercial 权重，公开数据 CC-BY 需署名 DOI。"],
        ],
        [4.0, 12.4],
        font_size=8.6,
    )

    heading(document, "七、一键命令", 1)
    code_block(document,
               "cd D:\\tzzcup\n"
               "python scripts/run_demo.py --mode quick                     # 2 秒离线全套结果\n"
               "python scripts/run_demo.py --mode photo --scene data/samples/synthetic_sem --max-views 4\n"
               "uvicorn api:app --host 127.0.0.1 --port 8001                # 本机接口\n"
               "streamlit run app.py                                        # 桌面演示\n"
               "$env:SLOPEWATCH_API_TOKEN=\"<32 位令牌>\"; uvicorn public_api:app --port 8011   # 远程入口")
    body(document, "产物默认在 demo_output/quick 或 demo_output/photo：report.pdf、surface.png、dod.png、atlas.png、metrics.json。",
         size=9.5, color=MUTED)

    heading(document, "八、彩排清单（前一天）", 1)
    for item in [
        "完整走一遍 5 分钟流程并计时；",
        "断网演练一次（切录屏）；",
        "GPU 关闭/占用演练一次（切 CPU 或预生成结果）；",
        "手机扫码测试（不同运营商网络）；",
        "打印 PDF 与二维码各 2 份；",
        "把第七节命令贴到桌面便签。",
    ]:
        bullet(document, item)

    heading(document, "九、演示后材料", 1)
    table(
        document,
        ["材料", "位置 / 说明"],
        [
            ["仓库二维码", "https://github.com/Zzz123bit/slopewatch"],
            ["一页纸", "定位 + 冲淤图 + 三个数字（体积/深度/冲沟数）"],
            ["PDF 报告样例", "docs/example_report.pdf"],
            ["技术选型清单", "docs/SlopeWatch_技术选型与框架清单.docx"],
            ["演示预生成结果", "docs/demo/quick、docs/demo/photo"],
        ],
        [4.0, 12.4],
        font_size=8.6,
    )

    body(document, "本手册与 scripts/run_demo.py、docs/DEMO_GUIDE.md 同步维护。", size=8.5, color=MUTED)
    document.save(str(OUT))
    print("saved", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    build()
