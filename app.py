from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from slopewatch.core import comparison_csv, comparison_json, compare_point_clouds, load_point_cloud, make_demo_comparison
from slopewatch.photogram import compare_reconstructions
from slopewatch.quality import inspect_images
from slopewatch.reconstruction import colmap_status, reconstruct_with_colmap
from slopewatch.report import build_report_pdf
from slopewatch.uploads import extract_image_archive
from slopewatch.vggt_recon import reconstruct_vggt, vggt_status
from slopewatch.visuals import render_dod_png, render_surface_png


st.set_page_config(page_title="坡镜智安", page_icon="⛰️", layout="wide")


def _small_grid(array: np.ndarray, maximum_side: int = 100):
    stride = max(1, int(np.ceil(max(array.shape) / maximum_side)))
    return array[::stride, ::stride]


def _comparison_figure(comparison):
    before = _small_grid(comparison.before_mm.copy())
    after = _small_grid(comparison.after_mm.copy())
    delta = _small_grid(comparison.delta_mm.copy())
    valid = _small_grid(comparison.valid)
    before[~valid] = np.nan
    after[~valid] = np.nan
    delta[~valid] = np.nan
    x = _small_grid(comparison.x_mm)[0, :]
    y = _small_grid(comparison.y_mm)[:, 0]

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=("基准时相", "变化后时相"),
        horizontal_spacing=0.03,
    )
    figure.add_trace(go.Surface(x=x, y=y, z=before, colorscale="Earth", showscale=False), row=1, col=1)
    figure.add_trace(go.Surface(x=x, y=y, z=after, colorscale="Earth", showscale=False), row=1, col=2)
    figure.update_layout(
        height=520,
        margin={"l": 0, "r": 0, "t": 45, "b": 0},
        scene={"xaxis_title": "X (mm)", "yaxis_title": "Y (mm)", "zaxis_title": "Z (mm)"},
        scene2={"xaxis_title": "X (mm)", "yaxis_title": "Y (mm)", "zaxis_title": "Z (mm)"},
    )
    delta_figure = go.Figure(
        go.Heatmap(
            x=x,
            y=y,
            z=delta,
            colorscale="RdBu",
            zmin=-max(1.0, float(np.nanmax(np.abs(delta)))),
            zmax=max(1.0, float(np.nanmax(np.abs(delta)))),
            colorbar={"title": "ΔZ (mm)"},
            hovertemplate="X %{x:.1f} mm<br>Y %{y:.1f} mm<br>ΔZ %{z:.2f} mm<extra></extra>",
        )
    )
    delta_figure.update_layout(
        height=460,
        margin={"l": 0, "r": 0, "t": 10, "b": 0},
        xaxis_title="X (mm)",
        yaxis_title="Y (mm)",
        yaxis={"scaleanchor": "x", "scaleratio": 1},
    )
    return figure, delta_figure


def _show_metrics(comparison):
    metrics = comparison.metrics
    cards = st.columns(4)
    cards[0].metric("冲刷体积（阈值以上）", f"{metrics['erosion_volume_l']:.4f} L")
    cards[1].metric("堆积体积（阈值以上）", f"{metrics['deposition_volume_l']:.4f} L")
    cards[2].metric("最大冲刷深度", f"{metrics['max_erosion_depth_mm']:.2f} mm")
    cards[3].metric("变化区域", f"{metrics['changed_area_m2']:.4f} m²")
    cards = st.columns(4)
    cards[0].metric("冲沟候选数", str(metrics["gully_count_est"]))
    cards[1].metric("最大堆积高度", f"{metrics['max_deposition_height_mm']:.2f} mm")
    rate = metrics["max_change_rate_mm_per_hour"]
    cards[2].metric("最大变化速率", "—" if rate is None else f"{rate:.2f} mm/h")
    cards[3].metric("有效网格", f"{metrics['valid_cells']:,}")


def _show_result(comparison, metadata: dict | None = None):
    _show_metrics(comparison)
    surface_figure, heatmap_figure = _comparison_figure(comparison)
    st.plotly_chart(surface_figure, use_container_width=True)
    st.subheader("时相高程变化图")
    st.plotly_chart(heatmap_figure, use_container_width=True)
    st.caption("蓝色区域表示高程降低，红色区域表示高程增加。体积只累计超过阈值的变化，raw 指标提供未滤波积分。冲沟长宽是候选区域几何估计。")
    if comparison.gullies:
        st.subheader("冲刷区候选")
        st.dataframe(comparison.gullies, use_container_width=True, hide_index=True)
    else:
        st.info("当前阈值下没有提取到冲刷区候选。")
    downloads = st.columns(3)
    downloads[0].download_button(
        "下载高程差 CSV", comparison_csv(comparison), file_name="slopewatch_change.csv", mime="text/csv", use_container_width=True
    )
    downloads[1].download_button(
        "下载指标 JSON", comparison_json(comparison, metadata), file_name="slopewatch_metrics.json", mime="application/json", use_container_width=True
    )
    downloads[2].download_button(
        "下载巡检 PDF",
        build_report_pdf(comparison.metrics, comparison.gullies, metadata),
        file_name="slopewatch_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )


st.title("坡镜智安")
st.caption("室内坡面多时相三维形貌分析原型")

with st.sidebar:
    page = st.radio("功能", ["演示样例", "点云差分", "防护效果", "照片质检", "照片重建", "照片对比（VGGT）"])
    colmap = colmap_status()
    vggt = vggt_status()
    st.divider()
    st.write("COLMAP", "可用" if colmap["available"] else "未配置")
    st.write("VGGT", "可用" if vggt["available"] else "未就绪")
    st.caption("接口文档地址为运行 API 后打开 /docs")


if page == "演示样例":
    st.info("当前为程序生成的示例坡面，用来演示差分、指标和报告下载链路，不代表真实试验数据。")
    if st.button("生成三维变化样例", type="primary") or "demo_comparison" not in st.session_state:
        st.session_state["demo_comparison"] = make_demo_comparison()
    _show_result(
        st.session_state["demo_comparison"],
        {"source": "synthetic_demo", "soil": "示例土", "rainfall_mm_h": 60, "protection": "无"},
    )

elif page == "点云差分":
    st.write("上传两个已经处于同一尺度、同一坐标系的单层地表点云。支持 ASCII 或二进制 PLY、XYZ、TXT、CSV。系统按 XYZ 的 XY 平面高度场插值，不能代表陡直侧壁。")
    with st.form("comparison_form"):
        columns = st.columns(2)
        before_file = columns[0].file_uploader("基准时相点云", type=["ply", "xyz", "txt", "csv"])
        after_file = columns[1].file_uploader("变化后时相点云", type=["ply", "xyz", "txt", "csv"])
        columns = st.columns(4)
        unit = columns[0].selectbox("点云坐标单位", ["mm", "m"])
        grid_step = columns[1].number_input("网格间距 (mm)", min_value=0.2, max_value=100.0, value=2.0, step=0.2)
        threshold = columns[2].number_input("变化阈值 (mm)", min_value=0.0, max_value=100.0, value=3.0, step=0.5)
        elapsed = columns[3].number_input("时相间隔 (小时)", min_value=0.0, max_value=10000.0, value=1.0, step=0.5)
        metadata_text = st.text_area(
            "试验记录 JSON（可选）",
            value='{"soil": "", "slope_deg": null, "rainfall_mm_h": null, "duration_min": null, "protection": "", "timepoint": ""}',
            height=100,
        )
        submitted = st.form_submit_button("开始差分分析", type="primary")
    if submitted:
        if before_file is None or after_file is None:
            st.error("请先上传两期点云。")
        else:
            try:
                metadata = json.loads(metadata_text)
                if not isinstance(metadata, dict):
                    raise ValueError("试验记录须为 JSON 对象")
                before = load_point_cloud(before_file.getvalue(), before_file.name, unit)
                after = load_point_cloud(after_file.getvalue(), after_file.name, unit)
                st.session_state["uploaded_comparison"] = compare_point_clouds(
                    before,
                    after,
                    grid_step_mm=grid_step,
                    erosion_threshold_mm=threshold,
                    elapsed_hours=elapsed,
                )
                st.session_state["uploaded_metadata"] = metadata
                st.success("差分分析完成。")
            except Exception as error:
                st.error(str(error))
    if "uploaded_comparison" in st.session_state:
        _show_result(st.session_state["uploaded_comparison"], st.session_state.get("uploaded_metadata", {}))

elif page == "防护效果":
    st.write("上传各组点云差分页面导出的指标 JSON，选择裸土对照，比较不同防护条件下的冲刷体积。")
    metric_files = st.file_uploader(
        "试验组指标 JSON",
        type=["json"],
        accept_multiple_files=True,
        key="protection_metrics_files",
    )
    trials = []
    parse_errors = []
    for file_index, metric_file in enumerate(metric_files or [], start=1):
        try:
            payload = json.loads(metric_file.getvalue())
            metrics = payload["metrics"]
            metadata = payload.get("metadata", {})
            if "erosion_volume_l" not in metrics:
                raise ValueError("缺少 erosion_volume_l 指标")
            trials.append(
                {
                    "样本": f"{file_index}. {metric_file.name}",
                    "土质": metadata.get("soil", ""),
                    "坡度": metadata.get("slope_deg", ""),
                    "降雨强度": metadata.get("rainfall_mm_h", ""),
                    "防护措施": metadata.get("protection", "未填写") or "未填写",
                    "冲刷体积 (L)": float(metrics["erosion_volume_l"]),
                }
            )
        except (ValueError, KeyError, TypeError) as error:
            parse_errors.append(f"{metric_file.name}: {error}")
    for error in parse_errors:
        st.warning(error)

    if len(trials) < 2:
        st.info("至少上传两个试验组的指标 JSON。建议各组使用相同土质、坡度、降雨条件和数据处理阈值。")
    else:
        control_options = [trial["样本"] for trial in trials]
        control_name = st.selectbox("裸土对照样本", control_options)
        control = next(trial for trial in trials if trial["样本"] == control_name)
        control_volume = control["冲刷体积 (L)"]
        comparison_rows = []
        for trial in trials:
            reduction = None
            if control_volume > 0:
                reduction = (control_volume - trial["冲刷体积 (L)"]) / control_volume * 100.0
            comparison_rows.append(
                {
                    **trial,
                    "相对裸土变化率 (%)": None if reduction is None else round(reduction, 2),
                    "样本类型": "裸土对照" if trial["样本"] == control_name else "防护组",
                }
            )
        st.dataframe(comparison_rows, use_container_width=True, hide_index=True)
        chart = go.Figure(
            go.Bar(
                x=[trial["防护措施"] for trial in trials],
                y=[trial["冲刷体积 (L)"] for trial in trials],
                text=[f"{trial['冲刷体积 (L)']:.4f} L" for trial in trials],
                textposition="outside",
                marker_color=["#D26B4B" if trial["样本"] == control_name else "#287C69" for trial in trials],
                hovertemplate="%{x}<br>冲刷体积 %{y:.5f} L<extra></extra>",
            )
        )
        chart.update_layout(
            height=420,
            xaxis_title="防护措施",
            yaxis_title="冲刷体积（阈值以上，L）",
            margin={"l": 20, "r": 20, "t": 25, "b": 45},
        )
        st.plotly_chart(chart, use_container_width=True)
        st.caption("正变化率表示相对裸土冲刷体积减少。实验条件不一致或样本数较少时，结果仅作描述性比较。")

elif page == "照片质检":
    st.write("检查照片分辨率、清晰度、曝光和可识别的 ArUco 标志。每个时相建议沿固定路径拍摄 25–40 张，并保持较高重叠。")
    uploads = st.file_uploader("选择一个时相的照片", type=["jpg", "jpeg", "png", "tif", "tiff", "webp"], accept_multiple_files=True)
    if st.button("检查照片", type="primary", disabled=not uploads):
        try:
            st.session_state["photo_quality"] = inspect_images([(item.name, item.getvalue()) for item in uploads])
        except Exception as error:
            st.error(str(error))
    if "photo_quality" in st.session_state:
        result = st.session_state["photo_quality"]
        st.metric("照片数量", result["image_count"])
        st.metric("有提示的照片", result["warning_count"])
        st.dataframe(result["images"], use_container_width=True, hide_index=True)

elif page == "照片重建":
    st.write("上传一个时相的照片 ZIP，使用本机 COLMAP 执行稀疏重建、稠密匹配并导出 PLY 点云。")
    if not colmap["available"]:
        st.warning("当前工作区没有检测到 COLMAP。先安装 COLMAP 并配置 PATH 或 COLMAP_EXE 环境变量。点云差分和演示页面不依赖 COLMAP。")
    archive = st.file_uploader("照片 ZIP", type=["zip"])
    image_size = st.slider("重建最大图像边长", min_value=512, max_value=2400, value=1200, step=128)
    if st.button("开始照片重建", type="primary", disabled=archive is None or not colmap["available"]):
        try:
            with tempfile.TemporaryDirectory(prefix="slopewatch-") as temporary:
                root = Path(temporary)
                archive_data = archive.getvalue()
                if len(archive_data) > 300 * 1024 * 1024:
                    raise ValueError("照片 ZIP 不能超过 300 MB")
                images_dir = root / "images"
                image_paths = extract_image_archive(archive_data, images_dir)
                st.session_state["reconstruction_quality"] = inspect_images([(path.name, path) for path in image_paths])
                cloud_path = reconstruct_with_colmap(images_dir, root / "output", image_size)
                st.session_state["reconstructed_cloud"] = cloud_path.read_bytes()
            st.success("重建完成。导出的点云仍需使用标尺恢复物理尺度，并与其他时相统一坐标系。")
        except Exception as error:
            st.error(str(error))
    if "reconstruction_quality" in st.session_state:
        st.dataframe(st.session_state["reconstruction_quality"]["images"], use_container_width=True, hide_index=True)
    if "reconstructed_cloud" in st.session_state:
        st.download_button("下载 COLMAP 点云 PLY", st.session_state["reconstructed_cloud"], file_name="fused.ply", mime="application/octet-stream")

elif page == "照片对比（VGGT）":
    st.write("上传雨前、雨后各一组多视角照片（ZIP），用 VGGT-1B 重建两期坡面，自动配准后计算冲淤。")
    if not vggt["available"]:
        st.warning(f"VGGT 当前不可用：{vggt['detail']}")
    left, right = st.columns(2)
    before_zip = left.file_uploader("雨前照片 ZIP（8–32 张）", type=["zip"], key="vggt_before")
    after_zip = right.file_uploader("雨后照片 ZIP（8–32 张）", type=["zip"], key="vggt_after")
    columns = st.columns(5)
    grid = columns[0].number_input("网格间距 (mm)", min_value=0.2, max_value=100.0, value=2.0, step=0.2, key="vggt_grid")
    threshold = columns[1].number_input("变化阈值 (mm)", min_value=0.0, max_value=100.0, value=3.0, step=0.5, key="vggt_threshold")
    elapsed = columns[2].number_input("时相间隔 (h)", min_value=0.0, max_value=10000.0, value=1.0, step=0.5, key="vggt_elapsed")
    scale = columns[3].number_input("尺度系数 (mm/单位)", min_value=0.000001, max_value=1000000.0, value=1.0,
                                    step=0.001, format="%.6f", key="vggt_scale")
    views = columns[4].selectbox("每期视角", [6, 8, 10, 12], index=1, key="vggt_views")
    if st.button("开始 VGGT 重建与对比", type="primary",
                 disabled=before_zip is None or after_zip is None or not vggt["available"]):
        try:
            with tempfile.TemporaryDirectory(prefix="slopewatch-vggt-") as temporary:
                root = Path(temporary)
                before_images = extract_image_archive(before_zip.getvalue(), root / "before")
                after_images = extract_image_archive(after_zip.getvalue(), root / "after")
                with st.status("VGGT 重建中…", expanded=True) as status_box:
                    st.write(f"雨前时相：{len(before_images)} 张照片")
                    before_cloud = reconstruct_vggt(before_images, max_views=views)
                    st.write(f"雨后时相：{len(after_images)} 张照片")
                    after_cloud = reconstruct_vggt(after_images, max_views=views)
                    status_box.update(label="相似变换配准与冲淤计算", state="running")
                    comparison, diagnostics = compare_reconstructions(
                        before_cloud.points,
                        after_cloud.points,
                        grid_step_mm=grid,
                        threshold_mm=threshold,
                        elapsed_hours=elapsed,
                        scale_mm_per_unit=scale,
                    )
                    status_box.update(label="分析完成", state="complete")
                st.session_state["vggt_comparison"] = comparison
                st.session_state["vggt_diagnostics"] = {
                    **diagnostics,
                    "photo": {"before": len(before_images), "after": len(after_images)},
                    "scale_mm_per_unit": scale,
                }
        except Exception as error:
            st.error(str(error))
    if "vggt_comparison" in st.session_state:
        comparison = st.session_state["vggt_comparison"]
        diagnostics = st.session_state.get("vggt_diagnostics", {})
        if not diagnostics.get("scale_calibrated", False):
            st.info("尺度系数仍为 1：深度与体积按重建单位输出，现场有标尺时请填写换算系数后重算。")
        st.image(render_surface_png(comparison), caption="雨前 / 雨后三维坡面")
        st.image(render_dod_png(comparison), caption="冲淤高程差图")
        _show_result(
            comparison,
            {
                "source": "vggt_photo",
                "scale_mm_per_unit": diagnostics.get("scale_mm_per_unit", 1.0),
                "photos": diagnostics.get("photo", {}),
            },
        )

