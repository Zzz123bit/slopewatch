# 坡镜智安 MVP

用于室内小型边坡试验的本地演示原型，包含合成样例、点云时相差分、照片质量检查、可选 COLMAP 重建和 FastAPI 任务接口。

## 校赛远程演示

GitHub 保存源码。比赛电脑运行 FastAPI 并保存任务结果。评委在手机或电脑上打开 HTTPS 地址，输入团队令牌，就能提交合成样例或两期点云、查看冲淤指标并下载巡检报告。远程页面与 API 共用一个域名。

```text
评委浏览器或调用程序 → Cloudflare HTTPS 地址 → Cloudflare Tunnel
                                                ↓
                      比赛电脑 127.0.0.1:8011 → 分析任务和报告
```

### 启动临时演示

在 Windows 比赛电脑上准备 Python 和 cloudflared。首次使用先安装下方“启动”中的 Python 依赖。安装 cloudflared 的命令为 `winget install --id Cloudflare.cloudflared --exact`，也可使用 Cloudflare 官方的 Windows 发行包。[cloudflared 下载](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/downloads/)

在项目根目录执行以下两条命令。第一条将本机的演示令牌复制到剪贴板，可单独发给评委。第二条启动服务并显示 `https://….trycloudflare.com`。保持比赛电脑联网、唤醒，保留该终端；按 Ctrl+C 结束。每次重新启动临时隧道都可能获得新地址。

```powershell
.\scripts\copy-demo-token.ps1
.\scripts\start-public.ps1
```

浏览器打开终端给出的地址，粘贴令牌并点击“运行样例”。切换到“上传点云”可以提交自己的数据。手机关闭 Wi-Fi、用移动网络尝试下载巡检 PDF，可以检验比赛现场的外网访问。网络中断后，页面提供对当前任务重新查询的按钮。批量照片可在 `/docs` 上传 ZIP，依赖比赛电脑配置的 COLMAP。

公网启动脚本使用单独的应用 `public_api:app`。本机开发入口 8001 和 Streamlit 8501 可继续使用。隧道的源地址固定为 `http://127.0.0.1:8011`。已有令牌可通过进程环境变量 `SLOPEWATCH_API_TOKEN` 传入（至少 32 位，由英文字母、数字、下划线、连字符组成）。所有持有团队令牌的人属于同一演示团队，能查询已知任务 ID 的结果。演示用的共享凭据适合邀请制校赛。

### 调用接口

远程公共入口 `GET /livez` 返回服务标识，`/docs` 可使用 **Authorize** 试调。所有 `/api/v1/` 请求，包括下载，均发送 `Authorization: Bearer <团队令牌>`。下表中的路径都相对于隧道 HTTPS 地址。

| 方法与路径 | 输入 | 响应 |
| --- | --- | --- |
| `POST /api/v1/jobs/demo` | 空请求 | `202`，含 `job_id`、`status_url` |
| `POST /api/v1/jobs/compare` | multipart 文件 `before`、`after`；可附 `unit`、`grid_step_mm` 等 | `202`，含任务 ID |
| `POST /api/v1/reconstructions` | multipart 文件 `photos_zip`；可附 `max_image_size` | `202`，COLMAP 未就绪时为 `503` |
| `GET /api/v1/jobs/{id}` | 任务 ID | `queued / running / complete / failed`、百分比进度 |
| `GET /api/v1/jobs/{id}/result` | 已完成任务 | 指标和导出文件名；任务未完成时为 `409` |
| `GET /api/v1/jobs/{id}/files/{name}` | 结果中列出的导出文件名 | CSV、JSON、PDF 或重建点云 |

以下 PowerShell 示例从另一台电脑运行。令牌通过隐藏输入读取。将 HTTPS 地址替换为终端显示的实际地址。

```powershell
$base = 'https://YOUR-DEMO.trycloudflare.com'
$secret = Read-Host '团队令牌' -AsSecureString
$headers = @{ Authorization = 'Bearer ' + ([Net.NetworkCredential]::new('', $secret)).Password }
$job = Invoke-RestMethod -Method Post -Uri "$base/api/v1/jobs/demo" -Headers $headers
$statusUrl = $base + $job.status_url
do {
    Start-Sleep -Seconds 1
    $state = Invoke-RestMethod -Uri $statusUrl -Headers $headers
} while ($state.status -in 'queued', 'running')
if ($state.status -ne 'complete') { throw $state.error }
Invoke-RestMethod -Uri "$statusUrl/result" -Headers $headers
Invoke-WebRequest -Uri "$statusUrl/files/report.pdf" -Headers $headers -OutFile report.pdf
```

如果上传返回 `413`，请按该 Cloudflare 套餐的单请求体积上限减小点云或照片 ZIP；同一次表单提交的两份点云合并计算。无效令牌返回 `401`，断开隧道后本机的任务结果仍保存在 `data/jobs/`。[Cloudflare 请求大小说明](https://developers.cloudflare.com/support/troubleshooting/http-status-codes/4xx-client-error/error-413/)

### 比赛当天使用固定域名

有 Cloudflare 托管的域名时，在控制台创建命名隧道，按提示在比赛电脑启动该隧道的 connector，添加一条 **Published application**，例如将 `demo.你的域名` 的 HTTP 服务指向 `http://127.0.0.1:8011`。比赛电脑从项目根目录运行 `.\scripts\start-public.ps1 -LocalOnly`。之后可以提前把固定的 HTTPS 链接印在展板上。Cloudflare connector 凭据和评委使用的 API 令牌分别保存。[Cloudflare 命名隧道步骤](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/get-started/create-remote-tunnel/) · [临时隧道说明](https://developers.cloudflare.com/cloudflare-one/networks/connectors/cloudflare-tunnel/do-more-with-tunnels/trycloudflare/)

## 启动

建议使用 Python 3.10 或 3.11 建立独立虚拟环境。

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py --server.address 127.0.0.1
```

浏览器默认打开 `http://localhost:8501`。另一终端启动 REST API：

```powershell
uvicorn api:app --host 127.0.0.1 --port 8001
```

接口文档位于 `http://127.0.0.1:8001/docs`，健康状态位于 `/api/v1/health`。如 8001 端口不可用，可改用其他端口。当前工作区的 Python 3.12 环境也可用于开发和测试。

## 页面功能

- **演示样例** 生成可重复的合成冲刷/堆积表面，展示三维面、变化热力图和 CSV、JSON、PDF 导出。
- **点云差分** 导入两个时相的 PLY、XYZ、TXT 或 CSV 点云，插值为共同高程网格并计算深度、面积、体积、变化速率和冲刷区候选。
- **防护效果** 汇总不同试验组导出的指标 JSON，选择裸土对照并计算各防护组冲刷体积变化率。
- **照片质检** 统计分辨率、清晰度、曝光和可见的 ArUco 标志数量。
- **照片重建** 配置了 COLMAP 后可从单时相照片 ZIP 生成稠密 PLY 点云。

## 点云输入要求

输入点云应表示单层地表高程，并使用相同尺度、相同坐标方向和共同的 X/Y 平面范围。单位可选 mm 或 m。推荐在坡面外围设置固定 ArUco/ChArUco 标志和已知长度标尺，将各时相转换到统一的物理坐标系。陡直侧壁在同一 XY 位置有多个 Z 值，不适用当前高度场差分。变化体积按超过阈值的高程差乘网格面积积分，单位以升输出。raw 体积会包含阈值以下变化，用于检查测量噪声。

CSV 首行可使用 `x,y,z` 列名。XYZ/TXT 每行默认前三列为 XYZ，COLMAP `points3D.txt` 会读取第 2–4 列。PLY 支持 ASCII 和 binary little/big endian，顶点需包含 `x`、`y`、`z` 属性。

冲刷区候选以变化阈值和连通区域计算。插值有效区域会受点云密度限制，稀疏覆盖区域不用于差分。冲沟长度、宽度是连通区域的主轴估计，用于原型展示，须用人工测量验证。程序不会把单目重建输出自动视为具有物理尺度的测量数据。

## COLMAP

照片重建需要额外安装 COLMAP。将 `colmap.exe`/`colmap` 可执行文件放入 `PATH`，或设置 `COLMAP_EXE` 指向可执行文件（批处理 `.bat` 不支持）。每个 ZIP 代表一个时相，建议包含 25–40 张有重叠的清晰照片。每个 ZIP 最多 300 张图片、展开后最多 1 GB、单图最多 50 MB，并限制单图像素数为 8000 万。Windows 上可先用 8–16 张低分辨率测试图验证完整链路。

COLMAP 结果使用自由尺度坐标。恢复已知长度比例和对齐到跨时相坐标系后，再将点云导入“点云差分”。VGGT 尚未接入本 MVP。

## REST API

- `GET /api/v1/health` 查看服务与 COLMAP 状态。
- `POST /api/v1/jobs/demo` 运行合成演示任务。
- `POST /api/v1/jobs/compare` 上传 `before` 和 `after` 点云，表单字段支持 `unit`、`grid_step_mm`、`erosion_threshold_mm`、`elapsed_hours`、`metadata_json`。
- `POST /api/v1/reconstructions` 上传 `photos_zip` 和 `max_image_size`，提交 COLMAP 照片重建。
- `GET /api/v1/jobs/{job_id}` 查询状态。
- `GET /api/v1/jobs/{job_id}/result` 读取指标结果。
- `GET /api/v1/jobs/{job_id}/files/{filename}` 下载 CSV、JSON、PDF 或重建点云。

任务数据保存在工作区 `data/jobs/`。接口默认只监听本机，适合本地演示。

## 测试

```powershell
python -m pytest -q
```

