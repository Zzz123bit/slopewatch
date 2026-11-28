# 坡镜智安 SlopeWatch · 项目现状与计划

> 版本 v1.0 · 2026-11-28 · 仓库提交 `64cbbad`
> 完整排版版本见 [SlopeWatch_项目现状与计划.docx](SlopeWatch_项目现状与计划.docx)

## 一、项目定位

面向室内小型边坡降雨试验的低成本视觉监测原型：多视角照片 → VGGT 重建两期坡面 → 自动配准 →
冲淤量化 + 语义图谱 → 巡检报告；提供网页、REST API 与带令牌的远程入口。

**当前结论**：几何、配准、量化、语义、报告与网页链路均已跑通并有测试覆盖；下一阶段瓶颈在
**真实试验数据的标注与验收**、**物理尺度自动标定**两件事，而不是代码。

## 二、系统架构

| 层次 | 主要模块 | 职责 |
|---|---|---|
| 采集与质控 | `scripts/annotate.py`、`preannotate.py`、`slopewatch/uploads.py`、`quality.py` | 打包上传、清晰度/曝光检查、标注与预标注 |
| 几何重建 | `slopewatch/vggt_recon.py`、`reconstruction.py`（COLMAP 备选） | VGGT-1B 推理、视角选择、权重校验、GPU/CPU 回退 |
| 配准与基准 | `slopewatch/photogram.py`、`registration.py`、`survey.py` | Sim(3) ICP、坡面平面定向、稳健去趋势；控制点/ArUco 解算（进阶） |
| 语义融合 | `slopewatch/segnet.py`、`fusion.py`、`models/segmentation.onnx` | 五类分割、掩膜对齐、多视投票投影、分类统计 |
| 量化与出图 | `slopewatch/core.py`、`visuals.py` | DoD、体积/深度/冲沟指标、三维图、冲淤图、语义图谱 |
| 服务与报告 | `api.py`、`public_api.py`、`jobs.py`、`report.py`、`web/index.html`、`app.py` | 任务队列、鉴权、下载、PDF、网页与 Streamlit |

## 三、端到端流程（照片对比）

| 步骤 | 做什么 | 输入 | 输出 | 耗时 |
|---|---|---|---|---|
| 1 采集 | 每期 8–32 张（建议 25–40 张）高重叠照片，固定路径，含标尺/ArUco | 照片 | 两个 ZIP | 10–20 分钟 |
| 2 上传 | 网页「照片对比」或 `POST /api/v1/jobs/photo-compare` | ZIP + 参数 | job_id | 数秒 |
| 3 重建 | VGGT-1B 分别重建两期，显存不足回退 CPU | 照片 | 两期点图 | GPU ≈20 秒/期；CPU 1–2 分钟/期 |
| 4 配准 | Sim(3) ICP → 坡面平面转 Z-up → Huber 平面去趋势 | 两期点云 | 统一坐标点云 | 数秒 |
| 5 语义 | UNet 五类分割 → 几何对齐 → 多视投票投影 | 雨后照片 | 类别栅格 | 数秒 |
| 6 量化 | DoD、体积、最大深度、冲沟候选、分区统计 | 点云 + 类别 | metrics.json / change.csv | 数秒 |
| 7 出报告 | 渲染三维坡面/冲淤图/图谱，生成 PDF | 指标 | surface.png / dod.png / atlas.png / report.pdf | 数秒 |

## 四、完成度总览

| 模块 | 状态 | 证据 |
|---|---|---|
| VGGT 双时相重建 | ✅ | health 返回 `vggt.available=true` |
| 相似变换配准 | ✅ | 随机相似变换后体积误差约 −3% |
| 冲淤量化与指标 | ✅ | 体积/深度/面积/冲沟/速率均有测试 |
| 三维·冲淤·图谱出图 | ✅ | `docs/surface.png`、`dod.png`、`atlas.png` |
| PDF 巡检报告 | ✅ | `docs/example_report.pdf` |
| 五类语义分割 | ✅ | mIoU 0.72；任务 `semantic.source=model` |
| 标注与微调工具链 | ✅ | 预标注 + 画笔 + LabelMe 转换 + 微调脚本 |
| 网页与 REST API | ✅ | 三种任务模式、令牌、下载、隧道脚本 |
| 自动化测试 | ✅ | `pytest` 28 passed |
| 真实场景精度 | ⚠️ 待验证 | 目前只有合成数据与几何仿真的量化结果 |
| 物理尺度自动标定 | ⚠️ 部分 | 可手工填 mm/单位；ArUco 控制点未接入 |
| 语义真实域适配 | ⚠️ 部分 | 合成数据训练，真实照片域差距明显 |

## 五、关键指标

- 几何链路：已知 2.78 L → 恢复 2.70 L（−3%），最大深度一致。
- 真实 VGGT + 合成照片：真值 3.99 L → 恢复 1.50 L（约 38%），损失主要来自细沟分辨极限。
- 语义分割：mIoU **0.72**，像素准确率 0.96；各类 IoU 0.57 / 0.46 / 0.38 / 0.97 / 0.98。
- 耗时：GPU 两期 4–8 视角约 40 秒；CPU 回退 3–5 分钟。

## 六、已知限制

| 问题 | 原因 | 影响 | 应对 |
|---|---|---|---|
| 尺度依赖人工 | VGGT 输出任意尺度 | 体积随尺度三次方放大 | 接 ArUco 控制点或页面标尺工具 |
| 语义域差距 | 合成数据训练 | 真实照片误判 | 真实照片标注 100–200 张微调 |
| 细沟分辨不足 | 1 m 距离 + 深度平滑 | 5–10 cm 细沟被低估 | 拉近拍摄、增加重叠与纹理 |
| 配准假设强 | 全局相似变换 | 视角差大时失败或吸收变化 | 固定机位、靶标、残差复核 |
| 仅支持两期 | before/after 模型 | 无时序趋势 | 扩展多期任务 |
| 显存约束 | 8 GB 笔记本 GPU | 8 视角易 OOM | 默认 8 视角或云 GPU |
| 演示安全 | 共享令牌、本机 HTTP | 滥用风险 | 强随机令牌、限流与审计 |

## 七、待补清单

### P0（必须完成）
1. 采集并标注 100–200 张真实试验照片（四类各 ≥15 张）。
2. 微调分割模型：val mIoU ≥ 0.75，冲沟/坍塌 IoU ≥ 0.5。
3. 物理尺度自动标定（标尺长度换算或 ArUco 控制点解算）。
4. 两组试验端到端验收，冲淤量与报告可复现。

### P1（可信度）
5. 精度评估报告（与人工测量/TLS 对照）。
6. 多期支持与时序曲线。
7. 失败检测与回退（低纹理/模糊提示重拍或切换 COLMAP）。
8. CI 与数据/模型版本管理。

### P2（加分）
9. 云端部署、多租户、限流与审计。
10. 采集辅助 App（引导重叠/曝光/路径）。
11. 报告模板定制（校赛/水保规范）。

## 八、运行手册

```powershell
python -m pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8001
streamlit run app.py --server.address 127.0.0.1

# 标注与微调
python scripts/preannotate.py --images data\real_photos --masks data\real_masks
python scripts/annotate.py    --images data\real_photos --masks data\real_masks
python scripts/finetune_segmentation.py --images data\real_photos --masks data\real_masks --epochs 20

# 测试
python -m pytest -q
```

## 九、交付物

- 代码：GitHub `Zzz123bit/slopewatch`（`64cbbad`）
- 模型：`models/segmentation.onnx`、`models/segmentation.pt`
- 文档：`README.md`、`docs/annotation_guide.md`、本文件
- 示例：`docs/*.png`、`docs/example_report.pdf`

## 十、两周冲刺

| 时间 | 目标 | 交付 |
|---|---|---|
| 第 1–2 天 | 采集并预标注 100–200 张真实照片 | 掩膜数据集 + 标注报告 |
| 第 3–4 天 | 微调与评估 | 更新模型 + 指标记录 |
| 第 5–7 天 | 接入尺度标定 | 页面尺度输入或自动解算 |
| 第 8–10 天 | 两组端到端验收 | 验收记录 + 报告样例 |
| 第 11–12 天 | 演示彩排 | 比赛演示脚本 |
| 第 13–14 天 | 缓冲与发布 | tag v1.0 |
