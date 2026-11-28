# -*- coding: utf-8 -*-
"""Generate the SlopeWatch technology-stack document (DOCX)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt

from docx_style import (GREEN, MUTED, body, bullet, code_block, heading, new_document,
                        note, style_run, table)

DOCS = Path(__file__).resolve().parents[1] / "docs"
OUT = DOCS / "SlopeWatch_技术选型与框架清单.docx"


def module(document, title, rows, why, limits):
    heading(document, title, 2)
    for label, value in rows:
        paragraph = document.add_paragraph()
        paragraph.paragraph_format.space_after = Pt(2)
        style_run(paragraph.add_run(f"{label}："), size=10, bold=True)
        style_run(paragraph.add_run(value), size=10)
    body(document, f"选择理由：{why}", size=9.8)
    body(document, f"限制与替代：{limits}", size=9.8, color=MUTED)


def build():
    document = new_document()

    title = document.add_paragraph()
    title.paragraph_format.space_before = Pt(56)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(title.add_run("坡镜智安 SlopeWatch"), size=26, bold=True, color=GREEN)
    subtitle = document.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    style_run(subtitle.add_run("技术选型与框架清单"), size=15, color=MUTED)

    document.add_paragraph()
    for line in [
        "代码仓库：https://github.com/Zzz123bit/slopewatch",
        "适用版本：v1.0（2026-11-28）",
        "版本信息：取自当前运行环境实测，非文档推测",
    ]:
        paragraph = document.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        style_run(paragraph.add_run(line), size=10)
        paragraph.paragraph_format.space_after = Pt(3)
    document.add_page_break()

    # ---------------- 0 说明 ----------------
    heading(document, "一、文档说明", 1)
    body(document, "本文回答一个问题：这套系统里每个模块到底用了什么框架、什么模型、什么库，版本是多少，"
                   "为什么选它，以及将来什么情况下该换掉它。")
    body(document, "阅读方式：第二节是技术栈总览；第三节逐模块说明；第四节是依赖版本总表；第五节是许可与合规；第六节是评估过但未采用的技术；第七节是替换与升级路径；第八节是模块与源码对照；第九节是结论。")

    # ---------------- 1 总览 ----------------
    heading(document, "二、技术栈总览", 1)
    table(
        document,
        ["#", "模块", "采用框架 / 模型", "版本", "作用"],
        [
            ["1", "三维重建", "VGGT-1B（PyTorch）", "vggt 0.0.1 / torch 2.7.1", "多视角照片 → 相机位姿、深度、点图"],
            ["2", "重建备选", "COLMAP（可选）", "未安装", "SfM + MVS 稠密重建"],
            ["3", "两期配准", "自研 Sim(3) ICP（NumPy/SciPy）", "numpy 1.26.4 / scipy 1.13.1", "点云统一尺度与坐标系"],
            ["4", "冲淤量化", "自研 DoD / 格网插值（SciPy）", "scipy 1.13.1", "体积、深度、冲沟、变化速率"],
            ["5", "语义分割", "自研 U-Net（PyTorch）→ ONNX", "torch 2.7.1 / onnx 1.23.2", "五类像素分类"],
            ["6", "分割推理", "OpenCV DNN（cv2.dnn）", "opencv 4.10.0", "部署期零 PyTorch 依赖"],
            ["7", "语义融合", "自研多视投票（NumPy）", "numpy 1.26.4", "2D 类别投影到 DEM 网格"],
            ["8", "服务端", "FastAPI + Uvicorn + Starlette", "0.165.0 / 0.54.0 / 1.10.0", "任务接口、鉴权、文件下载"],
            ["9", "前端", "原生 HTML / CSS / JavaScript", "—", "上传、轮询、结果展示"],
            ["10", "桌面演示", "Streamlit + Plotly", "1.67.0 / 5.24.1", "本机交互式演示"],
            ["11", "出图", "Matplotlib（Agg）+ Pillow", "3.9.2 / 10.4.0", "服务端渲染 PNG"],
            ["12", "报告", "ReportLab", "4.5.1", "中文 PDF 巡检报告"],
            ["13", "标注", "OpenCV HighGUI + LabelMe 兼容", "opencv 4.10.0", "画笔标注与多边形转换"],
            ["14", "合成数据", "自研渲染（NumPy 泼溅 + OpenCV）", "numpy / opencv", "训练与测试数据"],
            ["15", "测试与部署", "pytest / Cloudflare Tunnel / Git", "7.4.4", "回归测试、公网演示、版本管理"],
        ],
        [0.9, 2.4, 5.4, 3.6, 4.6],
        font_size=8.5,
    )
    note(document, "说明：表中版本为当前比赛机环境实测版本；requirements.txt 中给出的是可接受范围，部署时以锁定版本为准。")

    # ---------------- 2 逐模块 ----------------
    heading(document, "三、逐模块选型说明", 1)

    module(
        document, "3.1 三维重建：VGGT-1B",
        [("采用", "Meta 的 VGGT（Visual Geometry Grounded Transformer），1B 参数版本；PyTorch 实现，"
                  "权重 model.safetensors（4.68 GB），通过 safetensors 加载。"),
         ("入口", "slopewatch/vggt_recon.py；HTTP 任务 POST /api/v1/jobs/photo-compare")],
        "无需预先标定相机、无需 SfM 初始化，输入 2–24 张照片即可前馈输出位姿、深度、点图与置信度；"
        "对“基层随手拍”的非专业采集更友好，单卡即可运行，正好匹配低成本定位。",
        "输出是任意尺度，必须外部标定；8 GB 显存下 8 视角接近上限；对动态场景、弱纹理敏感。"
        "商用需换 VGGT-1B-Commercial 权重；备选是 COLMAP（精度高但慢、需安装、弱纹理更差）。",
    )

    module(
        document, "3.2 重建备选：COLMAP",
        [("采用", "COLMAP 命令行（feature_extractor / exhaustive_matcher / mapper / patch_match_stereo / stereo_fusion），"
                  "通过子进程调用。"),
         ("入口", "slopewatch/reconstruction.py；当前环境未安装，健康检查 colmap.available=false")],
        "经典 SfM + MVS 方案，在有纹理、需要高精度稠密点云时是可靠兜底；保留接口便于现场切换。",
        "需要单独安装、处理时间长、弱纹理坡面容易失败；因此只作为备选而非默认路径。",
    )

    module(
        document, "3.3 两期配准：自研 Sim(3) ICP",
        [("采用", "自研实现：SVD 求 Umeyama 相似变换 + KD 树最近邻迭代（SciPy cKDTree），"
                  "坡面主平面 PCA 转 Z-up，再做 Huber 平面去趋势。"),
         ("入口", "slopewatch/photogram.py；参数通过环境变量 SLOPEWATCH_ALIGN_INIT 控制初值")],
        "两期重建没有同名点，需要同时求尺度、旋转、平移；场景近似平面时普通 ICP 会滑移，"
        "因此加“平面定向 + 稳健去趋势”。不引入 Open3D/PCL，减少部署依赖。",
        "全局相似变换假设两期相机路径相近，视角差异大时会吸收真实变化；"
        "未来可换 Open3D 的 point-to-plane 或增加控制点约束。",
    )

    module(
        document, "3.4 冲淤量化：自研 DoD",
        [("采用", "网格高程差（DEM of Difference）：SciPy griddata 线性插值 + cKDTree 覆盖门限 + "
                  "ndimage.label 连通域提取冲沟。"),
         ("入口", "slopewatch/core.py（compare_point_clouds）；slopewatch/fusion.py（分类统计）")],
        "项目已有成熟的格网差分与指标代码，直接复用；不引入 GDAL/rasterio 等重型 GIS 依赖，"
        "体积按“阈值以上高程差 × 格网面积”积分，口径可解释。",
        "不适合陡直侧壁与多值表面；稀疏区域会被覆盖门限剔除；未来需要正射影像或断面分析时可加 GIS 工具。",
    )

    module(
        document, "3.5 语义分割：自研 U-Net + ONNX",
        [("采用", "4 层下采样/上采样的轻量 U-Net（base=24，5 类输出，256×256 输入），"
                  "PyTorch 训练、ONNX 导出；训练数据由脚本程序化生成。"),
         ("入口", "slopewatch/segnet.py（网络与推理类）、models/segmentation.onnx、"
                  "scripts/train_segmentation.py、scripts/finetune_segmentation.py")],
        "类别只有 5 个、样本可合成，轻量网络足够且能在 CPU 上 0.1–0.2 秒完成单张推理；"
        "ONNX 4.3 MB，方便部署与替换。",
        "合成数据与真实照片存在域差距（mIoU 0.72，真实场景会下降）；"
        "数据量到 500 张以上可换 SegFormer / DeepLabV3+，需要提示的分割可考虑 SAM2。",
    )

    module(
        document, "3.6 分割推理：OpenCV DNN（含 PyTorch 回退）",
        [("采用", "cv2.dnn.readNetFromONNX 加载模型，等距缩放 + 黑边填充 + 双线性还原，与训练预处理一致；"
                  "若 ONNX 缺失或损坏，自动回退到 PyTorch checkpoint。"),
         ("入口", "slopewatch/fusion.py（load_segmenter / segmentation_status）")],
        "部署机不需要安装 PyTorch 也能跑分割；OpenCV 已是项目依赖，零额外成本；"
        "回退机制让“刚微调完、还没导出 ONNX”的状态也能直接使用。",
        "cv2.dnn 对算子支持有限，换更大模型（如 SegFormer）时可能需要 ONNX Runtime 或 TensorRT。",
    )

    module(
        document, "3.7 语义融合：自研多视投票",
        [("采用", "把每个视角的类别掩膜按 VGGT 预处理几何对齐，再用对应的三维点投影到 DEM 网格，"
                  "逐视角多数投票，冲突或未覆盖记为 255。"),
         ("入口", "slopewatch/fusion.py（project_label_views / analyze_fusion / atlas_png）")],
        "单视角会有遮挡与透视误差，多视投票能显著降低边界噪声；"
        "实现只用 NumPy，不依赖任何 3D 深度学习库。",
        "投影精度受点图与掩膜对齐误差影响；类别极不均衡时投票会偏向多数类，必要时加类别权重。",
    )

    module(
        document, "3.8 服务端：FastAPI + Uvicorn + Starlette + Pydantic",
        [("采用", "FastAPI 提供 OpenAPI 与表单校验；Starlette 中间件做 Bearer 令牌鉴权；"
                  "Pydantic 校验 survey.py 的试验包配置；任务用 BackgroundTasks 后台执行。"),
         ("入口", "api.py（本机接口）、public_api.py（带令牌的公网入口）、slopewatch/jobs.py（任务）")],
        "轻量、类型安全、自带 /docs 便于评委与队友调试；后台任务足以支撑单机演示，无需引入 Celery/Redis。",
        "单进程后台线程不适合高并发；正式部署应换成任务队列（RQ/Celery）并加限流。",
    )

    module(
        document, "3.9 前端：原生 JavaScript + Streamlit",
        [("采用", "web/index.html 为单文件原生 HTML/CSS/JS（fetch + 1.5 秒轮询 + Blob 下载），无构建链；"
                  "app.py 用 Streamlit 做本机演示。"),
         ("入口", "web/index.html、app.py")],
        "比赛机可能离线或网络受限，单文件前端不依赖 CDN 与打包工具，拷贝即可用；"
        "Streamlit 让本地调试与图表交互更省事。",
        "原生 JS 维护成本随功能增长；若要复杂交互可迁移到 Vue/React，但会增加构建与部署复杂度。",
    )

    module(
        document, "3.10 出图：Matplotlib + Pillow + Plotly",
        [("采用", "服务端用 Matplotlib（Agg 后端）渲染三维坡面与冲淤热图，Pillow 合成语义图谱；"
                  "Streamlit 内用 Plotly 提供可旋转的三维视图。"),
         ("入口", "slopewatch/visuals.py、slopewatch/fusion.py（atlas_png）、app.py")],
        "服务端出图不依赖浏览器 WebGL 或 CDN，离线也能生成报告插图；"
        "Plotly 只用于本机交互，二者分工明确。",
        "Matplotlib 三维渲染较慢且无交互；需要交互式网页三维时可引入 Plotly.js 或 Three.js。",
    )

    module(
        document, "3.11 报告：ReportLab",
        [("采用", "ReportLab 生成 A4 PDF，注册微软雅黑（失败时回退 STSong-Light），表格排版指标与冲沟列表。"),
         ("入口", "slopewatch/report.py；示例 docs/example_report.pdf")],
        "纯 Python、无需 Office 或浏览器，服务器可直接产出可打印报告；中文字体可控。",
        "复杂版式不如 HTML→PDF 灵活；需要更精美模板时可换 WeasyPrint（HTML/CSS 排版）。",
    )

    module(
        document, "3.12 标注工具：OpenCV GUI + LabelMe 兼容",
        [("采用", "OpenCV HighGUI 实现画笔标注（类别 0–4、橡皮、撤销、自动保存）；"
                  "另提供 LabelMe JSON 多边形转掩膜脚本。"),
         ("入口", "scripts/annotate.py、scripts/preannotate.py、scripts/labelme_to_masks.py")],
        "零安装、零学习成本，队友打开就能改；预标注把“从零画”变成“改错”，200 张约 2 小时。",
        "只支持画笔与多边形，没有协同与版本管理；多人标注可换 CVAT / Label Studio。",
    )

    module(
        document, "3.13 合成数据：自研渲染",
        [("采用", "解析地形 + 高斯冲沟/坍塌 + 随机相机，NumPy 点云泼溅 + OpenCV 修复/形态学生成图像与掩膜；"
                  "同一套代码生成重建测试集与分割训练集。"),
         ("入口", "scripts/make_synthetic_photos.py、scripts/make_segmentation_dataset.py")],
        "没有真实标注数据时，合成数据可以先把链路、指标和模型跑通，并且自带真值（体积、类别掩膜）。",
        "与真实场景存在域差距，只能用于开发与回归，不能替代现场数据。",
    )

    module(
        document, "3.14 测试与部署：pytest + Cloudflare Tunnel + Git",
        [("采用", "pytest 覆盖核心、配准、报告、接口、分割与任务共 28 个用例；"
                  "Cloudflare Tunnel 把本机 8011 端口映射为 HTTPS 公网地址；Git/GitHub 管理代码。"),
         ("入口", "tests/、scripts/start-public.ps1、scripts/copy-demo-token.ps1")],
        "无公网 IP、无需改路由器即可让评委用手机访问；测试保证迭代不破坏既有链路。",
        "隧道地址每次可能变化（临时隧道），正式比赛建议用命名隧道 + 固定域名；"
        "公网暴露需配合强随机令牌与限流。",
    )

    # ---------------- 3 依赖版本 ----------------
    heading(document, "四、依赖与版本总表", 1)
    table(
        document,
        ["依赖", "版本", "用途", "许可"],
        [
            ["torch / torchvision", "2.7.1+cu126 / 0.22.1", "VGGT 推理、U-Net 训练", "BSD-3-Clause"],
            ["vggt", "0.0.1（源码安装）", "VGGT 模型定义与预处理", "VGGT License v1"],
            ["safetensors", "0.8.0", "权重加载", "Apache-2.0"],
            ["numpy", "1.26.4", "数值计算（锁定 1.x 以兼容 SciPy/VGGT）", "BSD-3-Clause"],
            ["scipy", "1.13.1", "插值、KD 树、连通域", "BSD-3-Clause"],
            ["opencv-python", "4.10.0", "图像处理、DNN 推理、标注 GUI", "Apache-2.0"],
            ["onnx", "1.23.2（仅训练环境）", "模型导出（与 protobuf 6 绑定）", "Apache-2.0"],
            ["fastapi / uvicorn", "0.165.0 / 0.54.0", "HTTP 接口", "MIT / BSD-3-Clause"],
            ["starlette / pydantic", "1.10.0 / 2.14.4", "中间件、数据校验", "BSD-3-Clause / MIT"],
            ["python-multipart", "0.0.34", "文件上传表单", "Apache-2.0"],
            ["streamlit / plotly", "1.67.0 / 5.24.1", "桌面演示与交互图", "Apache-2.0 / MIT"],
            ["matplotlib", "3.9.2", "服务端出图", "PSF / BSD 兼容"],
            ["pillow", "10.4.0", "图像读写与图谱合成", "HPND"],
            ["reportlab", "4.5.1", "PDF 报告", "BSD-3-Clause"],
            ["pytest / httpx", "7.4.4 / 0.27.0", "测试", "MIT / BSD-3-Clause"],
        ],
        [4.2, 3.4, 5.6, 3.4],
        font_size=8.6,
    )
    note(document, "环境注意：运行环境使用 numpy 1.26 + protobuf 5（Streamlit 要求）；"
                   "训练/导出环境需要 numpy≥2 + protobuf≥6 + onnx，建议独立虚拟环境，导出后把 ONNX 拷回。")

    # ---------------- 4 许可 ----------------
    heading(document, "五、许可与合规", 1)
    body(document, "外部模型与库的许可直接影响能否商用与参赛提交，逐项说明如下。", size=10)
    table(
        document,
        ["对象", "许可", "关键限制"],
        [
            ["VGGT 代码", "VGGT License v1（2025-07-29 更新）", "允许商用，排除军事用途；需保留许可与 AUP"],
            ["VGGT-1B 原始权重", "非商用", "研究/比赛可用，产品化不可直接使用"],
            ["VGGT-1B-Commercial 权重", "商用许可（gated）", "需在 Hugging Face 接受条款后下载并换用"],
            ["COLMAP（若启用）", "BSD-3-Clause", "宽松，可商用"],
            ["PyTorch / torchvision", "BSD-3-Clause", "宽松，可商用"],
            ["NumPy / SciPy / Matplotlib", "BSD 系", "宽松，可商用"],
            ["OpenCV", "Apache-2.0", "宽松，可商用（含专利授权条款）"],
            ["FastAPI / Uvicorn / Starlette / Pydantic", "MIT / BSD", "宽松，可商用"],
            ["Streamlit / Plotly", "Apache-2.0 / MIT", "宽松，可商用"],
            ["ReportLab / Pillow", "BSD / HPND", "宽松，可商用"],
            ["Pixabay 参考照片", "Pixabay Content License", "仅作演示参考，未参与训练"],
        ],
        [5.0, 5.2, 6.4],
        font_size=8.6,
    )
    body(document, "结论：当前演示与比赛用途没有许可障碍；一旦产品化，必须把 VGGT-1B 换成 "
                   "VGGT-1B-Commercial 权重，并在仓库保留许可文本。", size=10, bold=True)

    # ---------------- 5 未采用 ----------------
    heading(document, "六、评估过但未采用的技术", 1)
    table(
        document,
        ["技术", "为什么没有作为默认方案", "何时值得重新考虑"],
        [
            ["COLMAP 全流程", "需单独安装、处理慢、弱纹理坡面易失败", "有纹理、要求高精度、可接受分钟级处理"],
            ["VGGT-Ω", "需额外包与权重，控制点解算流程未接入", "需要控制点级精度或动态场景"],
            ["NeRF / 3DGS", "几何精度与绝对尺度不适合测量，训练慢", "只做可视化展示，不做量化"],
            ["SAM / SAM2", "类别无关、需要提示、模型大", "作为标注辅助工具，而不是在线分割器"],
            ["SegFormer / DeepLabV3+", "需要更多真实标注与训练资源", "真实数据超过 500 张后"],
            ["GDAL / rasterio / PDAL", "依赖重、部署复杂", "需要正射影像、点云滤波或断面分析"],
            ["Open3D / PCL", "增加 C++ 依赖，自研 ICP 已够用", "配准精度不足或需要点云可视化"],
            ["Celery / Redis", "单机演示不需要分布式队列", "并发任务或多人使用"],
        ],
        [3.8, 7.4, 5.4],
        font_size=8.6,
    )

    # ---------------- 6 升级路径 ----------------
    heading(document, "七、替换与升级路径", 1)
    table(
        document,
        ["模块", "当前", "候选", "触发条件"],
        [
            ["三维重建", "VGGT-1B（fp32，8 视角）", "VGGT-1B-Commercial / VGGT-Ω / 分块流式推理", "商用授权、长序列或精度不足"],
            ["绝对尺度", "手工填 mm/单位", "ArUco/AprilTag 控制点解算（survey.py 已有基础）", "需要真实体积而非相对值"],
            ["两期配准", "自研 Sim(3) + 平面去趋势", "Open3D point-to-plane / 控制点约束", "配准残差超限或视角差异大"],
            ["语义分割", "UNet 256 + ONNX", "SegFormer / DeepLabV3+ / SAM2 微调", "真实数据 ≥500 张或 mIoU 低于 0.75"],
            ["语义推理", "OpenCV DNN", "ONNX Runtime / TensorRT", "换更大模型或需要 GPU 加速"],
            ["服务端", "uvicorn 单进程", "gunicorn + uvicorn workers + Nginx + 限流", "并发用户超过 5 人"],
            ["任务存储", "本地 data/jobs", "MinIO / S3 对象存储", "多机部署或长期留存"],
            ["监控", "文件日志", "结构化日志 + Prometheus + 告警", "正式对外服务"],
        ],
        [2.8, 4.8, 4.8, 4.2],
        font_size=8.6,
    )

    # ---------------- 7 文件对照 ----------------
    heading(document, "八、模块与源码对照表", 1)
    table(
        document,
        ["模块", "主要文件", "说明"],
        [
            ["VGGT 重建", "slopewatch/vggt_recon.py", "权重校验、预处理几何、推理、视角选择"],
            ["COLMAP 备选", "slopewatch/reconstruction.py", "子进程封装五个 COLMAP 命令"],
            ["配准", "slopewatch/photogram.py、registration.py", "Sim(3) ICP、平面定向、去趋势、控制点变换"],
            ["量化", "slopewatch/core.py、fusion.py", "DoD、指标、分类统计、图谱"],
            ["分割模型", "slopewatch/segnet.py、models/segmentation.onnx", "U-Net 定义、训练/推理预处理"],
            ["出图与报告", "slopewatch/visuals.py、report.py", "三维图、冲淤图、PDF"],
            ["任务与服务", "slopewatch/jobs.py、api.py、public_api.py", "任务编排、接口、鉴权"],
            ["前端", "web/index.html、app.py", "网页与 Streamlit"],
            ["标注", "scripts/annotate.py、preannotate.py、labelme_to_masks.py", "画笔、预标注、格式转换"],
            ["训练", "scripts/train_segmentation.py、finetune_segmentation.py", "从零训练与真实数据微调"],
            ["合成数据", "scripts/make_synthetic_photos.py、make_segmentation_dataset.py", "重建测试集与分割训练集"],
            ["测试", "tests/（7 个文件，28 个用例）", "核心、配准、报告、接口、任务、分割"],
        ],
        [2.6, 7.6, 6.4],
        font_size=8.6,
    )

    # ---------------- 8 结论 ----------------
    heading(document, "九、结论", 1)
    body(document, "这套技术栈的核心取舍是“轻、可离线、可替换”：重建用 VGGT 换取免标定与速度，"
                   "配准与量化用自研 NumPy/SciPy 代码换取可控与零重依赖，语义用小型 U-Net + ONNX 换取部署简单，"
                   "服务端用 FastAPI 换取开发效率。")
    body(document, "两个需要长期跟进的点：一是 VGGT 的许可与权重版本（商用必须换 Commercial 权重）；"
                   "二是语义模型的数据来源（合成数据只能打通链路，真实场景必须用自己的照片微调）。", bold=False)
    body(document, "本文档与代码同步维护；涉及版本升级时，请同时更新第三节的版本号与第四节依赖表。", size=9, color=MUTED)

    document.save(str(OUT))
    print("saved", OUT, OUT.stat().st_size)


if __name__ == "__main__":
    build()
