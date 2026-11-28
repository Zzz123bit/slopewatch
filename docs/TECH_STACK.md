# 坡镜智安 SlopeWatch · 技术选型与框架清单

> 版本 v1.0 · 2026-11-28 · 仓库 <https://github.com/Zzz123bit/slopewatch>
> 排版版本：[SlopeWatch_技术选型与框架清单.docx](SlopeWatch_技术选型与框架清单.docx)

## 一、技术栈总览

| # | 模块 | 采用框架 / 模型 | 版本 | 作用 |
|---|---|---|---|---|
| 1 | 三维重建 | VGGT-1B（PyTorch） | vggt 0.0.1 / torch 2.7.1 | 多视角照片 → 位姿、深度、点图 |
| 2 | 重建备选 | COLMAP（可选） | 未安装 | SfM + MVS 稠密重建 |
| 3 | 两期配准 | 自研 Sim(3) ICP（NumPy/SciPy） | numpy 1.26.4 / scipy 1.13.1 | 统一尺度与坐标系 |
| 4 | 冲淤量化 | 自研 DoD / 格网插值（SciPy） | scipy 1.13.1 | 体积、深度、冲沟、速率 |
| 5 | 语义分割 | 自研 U-Net（PyTorch）→ ONNX | torch 2.7.1 / onnx 1.23.2 | 五类像素分类 |
| 6 | 分割推理 | OpenCV DNN（cv2.dnn） | opencv 4.10.0 | 部署期零 PyTorch 依赖 |
| 7 | 语义融合 | 自研多视投票（NumPy） | numpy 1.26.4 | 2D 类别 → DEM 网格 |
| 8 | 服务端 | FastAPI + Uvicorn + Starlette | 0.165 / 0.54 / 1.10 | 任务接口、鉴权、下载 |
| 9 | 前端 | 原生 HTML / CSS / JS | — | 上传、轮询、展示 |
| 10 | 桌面演示 | Streamlit + Plotly | 1.67 / 5.24 | 本机交互演示 |
| 11 | 出图 | Matplotlib（Agg）+ Pillow | 3.9 / 10.4 | 服务端渲染 PNG |
| 12 | 报告 | ReportLab | 4.5 | 中文 PDF 报告 |
| 13 | 标注 | OpenCV HighGUI + LabelMe 兼容 | opencv 4.10 | 画笔标注与格式转换 |
| 14 | 合成数据 | 自研渲染（NumPy 泼溅 + OpenCV） | — | 训练与测试数据 |
| 15 | 测试与部署 | pytest / Cloudflare Tunnel / Git | 7.4 | 回归测试、公网演示、版本管理 |

## 二、逐模块选型说明

### 2.1 三维重建：VGGT-1B

- **采用**：Meta VGGT（Visual Geometry Grounded Transformer）1B 版本，PyTorch，权重 `model.safetensors`（4.68 GB），safetensors 加载。
- **入口**：`slopewatch/vggt_recon.py`，任务 `POST /api/v1/jobs/photo-compare`。
- **理由**：免相机标定与 SfM 初始化，2–24 张照片前馈输出位姿/深度/点图/置信度；单卡可跑，适合“基层随手拍”。
- **限制/替代**：输出任意尺度，必须外部标定；8 GB 显存下 8 视角接近上限；动态场景与弱纹理敏感。商用需换 VGGT-1B-Commercial 权重；备选 COLMAP。

### 2.2 重建备选：COLMAP

- **采用**：COLMAP 命令行五步（feature_extractor / exhaustive_matcher / mapper / patch_match_stereo / stereo_fusion），子进程调用。
- **入口**：`slopewatch/reconstruction.py`；当前环境未安装，`colmap.available=false`。
- **理由**：有纹理、要高精度时的可靠兜底。
- **限制**：需安装、慢、弱纹理易失败，仅作备选。

### 2.3 两期配准：自研 Sim(3) ICP

- **采用**：SVD 求 Umeyama 相似变换 + KD 树最近邻迭代；坡面主平面 PCA 转 Z-up，Huber 平面去趋势。
- **入口**：`slopewatch/photogram.py`；初值由 `SLOPEWATCH_ALIGN_INIT` 控制。
- **理由**：两期无同名点，需同时求尺度/旋转/平移；平面场景普通 ICP 会滑移，故加平面定向与去趋势；不引入 Open3D/PCL。
- **限制**：全局相似变换假设相机路径相近，视角差异大时会吸收真实变化；可换 Open3D point-to-plane 或加控制点。

### 2.4 冲淤量化：自研 DoD

- **采用**：SciPy `griddata` 线性插值 + `cKDTree` 覆盖门限 + `ndimage.label` 连通域提取冲沟。
- **入口**：`slopewatch/core.py`（`compare_point_clouds`）、`slopewatch/fusion.py`（分类统计）。
- **理由**：复用已有格网差分代码，不引入 GDAL/rasterio；体积 = 阈值以上高程差 × 格网面积，口径可解释。
- **限制**：不适合陡直侧壁与多值表面；稀疏区被覆盖门限剔除。

### 2.5 语义分割：自研 U-Net + ONNX

- **采用**：4 层下/上采样轻量 U-Net（base=24，5 类，256×256），PyTorch 训练、ONNX 导出。
- **入口**：`slopewatch/segnet.py`、`models/segmentation.onnx`、`scripts/train_segmentation.py`、`scripts/finetune_segmentation.py`。
- **理由**：类别少、可合成数据，轻量网络足够，CPU 单张 0.1–0.2 秒；ONNX 4.3 MB，部署与替换方便。
- **限制**：合成数据与真实照片有域差距（mIoU 0.72）；数据 ≥500 张可换 SegFormer / DeepLabV3+，SAM2 可作标注辅助。

### 2.6 分割推理：OpenCV DNN（含 PyTorch 回退）

- **采用**：`cv2.dnn.readNetFromONNX`；等距缩放 + 黑边 + 双线性还原，与训练预处理一致；ONNX 缺失/损坏时回退 PyTorch checkpoint。
- **入口**：`slopewatch/fusion.py`（`load_segmenter` / `segmentation_status`）。
- **理由**：部署机无需 PyTorch；回退让“刚微调未导出”也能用。
- **限制**：cv2.dnn 算子支持有限，换大模型时可能需要 ONNX Runtime / TensorRT。

### 2.7 语义融合：自研多视投票

- **采用**：各视角掩膜按 VGGT 预处理几何对齐，三维点投影到 DEM 网格，逐视角多数投票，冲突记 255。
- **入口**：`slopewatch/fusion.py`（`project_label_views` / `analyze_fusion` / `atlas_png`）。
- **理由**：单视角有遮挡与透视误差，多视投票降噪；纯 NumPy，无需 3D 深度学习库。
- **限制**：投影精度受点图与掩膜对齐误差影响；类别不均衡时偏向多数类。

### 2.8 服务端：FastAPI + Uvicorn + Starlette + Pydantic

- **采用**：FastAPI（OpenAPI、表单校验）、Starlette 中间件（Bearer 鉴权）、Pydantic（survey 配置）、BackgroundTasks。
- **入口**：`api.py`、`public_api.py`、`slopewatch/jobs.py`。
- **理由**：轻量、类型安全、自带 `/docs`；单机演示无需 Celery/Redis。
- **限制**：单进程后台线程不适合高并发；正式部署换任务队列并限流。

### 2.9 前端：原生 JavaScript + Streamlit

- **采用**：`web/index.html` 单文件原生 HTML/CSS/JS（fetch + 1.5 秒轮询 + Blob 下载），无构建链；`app.py` 用 Streamlit。
- **理由**：比赛机可能离线，单文件不依赖 CDN 与打包；Streamlit 便于本地调试。
- **限制**：原生 JS 维护成本随功能增长；复杂交互可迁 Vue/React。

### 2.10 出图：Matplotlib + Pillow + Plotly

- **采用**：服务端 Matplotlib（Agg）渲染三维坡面与冲淤热图，Pillow 合成语义图谱；Streamlit 内 Plotly 提供可旋转三维视图。
- **入口**：`slopewatch/visuals.py`、`slopewatch/fusion.py`、`app.py`。
- **理由**：服务端出图不依赖 WebGL/CDN，离线可用；Plotly 仅用于本机交互。
- **限制**：Matplotlib 三维较慢且无交互；网页交互可上 Plotly.js / Three.js。

### 2.11 报告：ReportLab

- **采用**：ReportLab 生成 A4 PDF，注册微软雅黑（失败回退 STSong-Light）。
- **入口**：`slopewatch/report.py`；示例 `docs/example_report.pdf`。
- **理由**：纯 Python、无需 Office/浏览器；中文字体可控。
- **限制**：复杂版式不如 HTML→PDF；可换 WeasyPrint。

### 2.12 标注工具：OpenCV GUI + LabelMe 兼容

- **采用**：OpenCV HighGUI 画笔（类别 0–4、橡皮、撤销、自动保存）；LabelMe JSON 多边形转掩膜。
- **入口**：`scripts/annotate.py`、`scripts/preannotate.py`、`scripts/labelme_to_masks.py`。
- **理由**：零安装、零学习成本；预标注把“从零画”变成“改错”。
- **限制**：无协同与版本管理；多人标注可换 CVAT / Label Studio。

### 2.13 合成数据：自研渲染

- **采用**：解析地形 + 高斯冲沟/坍塌 + 随机相机，NumPy 点云泼溅 + OpenCV 修复/形态学。
- **入口**：`scripts/make_synthetic_photos.py`、`scripts/make_segmentation_dataset.py`。
- **理由**：无真实标注时先跑通链路，自带真值（体积、类别掩膜）。
- **限制**：域差距明显，只能用于开发与回归。

### 2.14 测试与部署：pytest + Cloudflare Tunnel + Git

- **采用**：pytest 28 个用例；Cloudflare Tunnel 映射本机 8011 为 HTTPS；Git/GitHub 管理。
- **入口**：`tests/`、`scripts/start-public.ps1`、`scripts/copy-demo-token.ps1`。
- **理由**：无公网 IP 即可手机访问；测试保证迭代不破坏链路。
- **限制**：临时隧道地址会变，正式比赛用命名隧道；公网需强令牌与限流。

## 三、依赖与版本

| 依赖 | 版本 | 用途 | 许可 |
|---|---|---|---|
| torch / torchvision | 2.7.1+cu126 / 0.22.1 | VGGT 推理、U-Net 训练 | BSD-3-Clause |
| vggt | 0.0.1（源码安装） | VGGT 模型与预处理 | VGGT License v1 |
| safetensors | 0.8.0 | 权重加载 | Apache-2.0 |
| numpy | 1.26.4 | 数值计算（锁 1.x 兼容 SciPy/VGGT） | BSD-3-Clause |
| scipy | 1.13.1 | 插值、KD 树、连通域 | BSD-3-Clause |
| opencv-python | 4.10.0 | 图像处理、DNN、标注 GUI | Apache-2.0 |
| onnx | 1.23.2（仅训练环境） | 模型导出 | Apache-2.0 |
| fastapi / uvicorn | 0.165 / 0.54 | HTTP 接口 | MIT / BSD |
| starlette / pydantic | 1.10 / 2.14 | 中间件、校验 | BSD / MIT |
| streamlit / plotly | 1.67 / 5.24 | 桌面演示与交互图 | Apache-2.0 / MIT |
| matplotlib | 3.9.2 | 服务端出图 | PSF/BSD |
| pillow | 10.4.0 | 图像读写 | HPND |
| reportlab | 4.5.1 | PDF 报告 | BSD-3-Clause |
| pytest / httpx | 7.4 / 0.27 | 测试 | MIT / BSD |

> 运行环境：numpy 1.26 + protobuf 5（Streamlit 要求）；训练/导出需要 numpy≥2 + protobuf≥6 + onnx，建议独立虚拟环境，导出后拷回 ONNX。

## 四、许可与合规

| 对象 | 许可 | 关键限制 |
|---|---|---|
| VGGT 代码 | VGGT License v1（2025-07-29） | 允许商用，排除军事；保留许可与 AUP |
| VGGT-1B 原始权重 | 非商用 | 研究/比赛可用，产品化不可直接使用 |
| VGGT-1B-Commercial | 商用许可（gated） | 需接受条款后换用 |
| COLMAP | BSD-3 | 可商用 |
| PyTorch / NumPy / SciPy | BSD 系 | 可商用 |
| OpenCV | Apache-2.0 | 可商用 |
| FastAPI / Uvicorn / Starlette / Pydantic | MIT / BSD | 可商用 |
| Streamlit / Plotly | Apache-2.0 / MIT | 可商用 |
| ReportLab / Pillow | BSD / HPND | 可商用 |
| Pixabay 参考照片 | Pixabay Content License | 仅演示，未参与训练 |

**结论**：当前比赛用途无许可障碍；产品化必须换 VGGT-1B-Commercial 权重并保留许可文本。

## 五、评估过但未采用

| 技术 | 未采用原因 | 何时重新考虑 |
|---|---|---|
| COLMAP 全流程 | 安装、慢、弱纹理差 | 有纹理、要求高精度 |
| VGGT-Ω | 额外包/权重，控制点流程未接 | 需要控制点级精度 |
| NeRF / 3DGS | 几何与尺度不适合测量 | 只做可视化 |
| SAM / SAM2 | 类别无关、需提示、模型大 | 标注辅助 |
| SegFormer / DeepLabV3+ | 需更多真实标注 | 数据 ≥500 张 |
| GDAL / rasterio / PDAL | 依赖重 | 正射影像/断面分析 |
| Open3D / PCL | 增加 C++ 依赖 | 配准精度不足 |
| Celery / Redis | 单机演示不需要 | 并发/多人 |

## 六、替换与升级路径

| 模块 | 当前 | 候选 | 触发条件 |
|---|---|---|---|
| 三维重建 | VGGT-1B（fp32，8 视角） | Commercial / VGGT-Ω / 分块流式 | 商用、长序列、精度不足 |
| 绝对尺度 | 手工 mm/单位 | ArUco/AprilTag 控制点解算 | 需要真实体积 |
| 两期配准 | 自研 Sim(3) + 去趋势 | Open3D point-to-plane / 控制点 | 残差超限、视角差大 |
| 语义分割 | UNet 256 + ONNX | SegFormer / DeepLabV3+ / SAM2 | 数据 ≥500 或 mIoU<0.75 |
| 语义推理 | OpenCV DNN | ONNX Runtime / TensorRT | 换大模型或需 GPU |
| 服务端 | uvicorn 单进程 | gunicorn workers + Nginx + 限流 | 并发 >5 |
| 任务存储 | 本地 data/jobs | MinIO / S3 | 多机、长期留存 |
| 监控 | 文件日志 | 结构化日志 + Prometheus | 正式对外 |

## 七、模块与源码对照

| 模块 | 主要文件 |
|---|---|
| VGGT 重建 | `slopewatch/vggt_recon.py` |
| COLMAP 备选 | `slopewatch/reconstruction.py` |
| 配准 | `slopewatch/photogram.py`、`registration.py` |
| 量化 | `slopewatch/core.py`、`fusion.py` |
| 分割模型 | `slopewatch/segnet.py`、`models/segmentation.onnx` |
| 出图与报告 | `slopewatch/visuals.py`、`report.py` |
| 任务与服务 | `slopewatch/jobs.py`、`api.py`、`public_api.py` |
| 前端 | `web/index.html`、`app.py` |
| 标注 | `scripts/annotate.py`、`preannotate.py`、`labelme_to_masks.py` |
| 训练 | `scripts/train_segmentation.py`、`finetune_segmentation.py` |
| 合成数据 | `scripts/make_synthetic_photos.py`、`make_segmentation_dataset.py` |
| 测试 | `tests/`（28 个用例） |

## 八、结论

取舍原则是“轻、可离线、可替换”：VGGT 换免标定与速度，配准/量化用 NumPy+SciPy 换可控与零重依赖，
小型 U-Net + ONNX 换部署简单，FastAPI 换开发效率。

需要长期跟进两点：**VGGT 的许可与权重版本**（商用必须换 Commercial）、**语义模型的数据来源**
（合成数据只打通链路，真实场景必须用现场照片微调）。
