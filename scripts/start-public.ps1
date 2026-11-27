[CmdletBinding()]
param(
    [switch]$LocalOnly,
    [ValidateSet('auto', 'http2', 'quic')][string]$Protocol = 'auto'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$origin = 'http://127.0.0.1:8011'
$python = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $python = (Get-Command python -CommandType Application -ErrorAction Stop | Select-Object -First 1).Source
}
if (-not $LocalOnly) {
    $installed = Get-Command cloudflared.exe -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    $connector = if ($installed) { $installed.Source } else { Join-Path $env:LOCALAPPDATA 'SlopeWatch/tools/cloudflared.exe' }
    if (-not (Test-Path -LiteralPath $connector -PathType Leaf)) {
        throw 'Install cloudflared (winget install --id Cloudflare.cloudflared --exact), or use -LocalOnly.'
    }
}
if (Get-NetTCPConnection -LocalPort 8011 -State Listen -ErrorAction SilentlyContinue) {
    throw 'TCP 8011 is in use. Stop the previous instance and retry.'
}

function Test-OwnedListener([int]$ProcessId, [int]$OwnerId) {
    # A Windows virtual environment may launch a child Python interpreter.
    while ($OwnerId -gt 0) {
        if ($ProcessId -eq $OwnerId) { return $true }
        $owner = Get-CimInstance Win32_Process -Filter "ProcessId = $OwnerId" -ErrorAction SilentlyContinue
        if (-not $owner -or $owner.ParentProcessId -eq $OwnerId) { break }
        $OwnerId = $owner.ParentProcessId
    }
    return $false
}

function Stop-OwnedTree($Process) {
    if ($Process -and -not $Process.HasExited) {
        & "$env:SystemRoot\System32\taskkill.exe" /PID $Process.Id /T /F | Out-Null
    }
}

$originalToken = $env:SLOPEWATCH_API_TOKEN
$logs = Join-Path ([IO.Path]::GetTempPath()) ('slopewatch-' + [guid]::NewGuid().ToString('N'))
$api = $null
$tunnel = $null
try {
    if ([string]::IsNullOrEmpty($originalToken)) {
        $cache = Join-Path $env:LOCALAPPDATA 'SlopeWatch/api-token.dpapi'
        $secret = if (Test-Path -LiteralPath $cache -PathType Leaf) {
            ConvertTo-SecureString ([IO.File]::ReadAllText($cache).Trim())
        } else {
            Read-Host 'API token (32+ characters of A-Z, a-z, 0-9, _ or -)' -AsSecureString
        }
        try {
            $env:SLOPEWATCH_API_TOKEN = [Net.NetworkCredential]::new('', $secret).Password
        } finally {
            $secret.Dispose()
        }
    }
    if ($env:SLOPEWATCH_API_TOKEN -cnotmatch '\A[A-Za-z0-9_-]{32,}\z') {
        throw 'Supply a URL-safe SLOPEWATCH_API_TOKEN of at least 32 characters.'
    }
    [IO.Directory]::CreateDirectory($logs) | Out-Null
    $apiLog = Join-Path $logs 'api.log'
    $api = Start-Process -FilePath $python -ArgumentList '-m','uvicorn','public_api:app','--host','127.0.0.1','--port','8011','--no-access-log' -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardError $apiLog -PassThru
    $env:SLOPEWATCH_API_TOKEN = $null

    $ready = $false
    $deadline = [datetime]::UtcNow.AddSeconds(60)
    while ([datetime]::UtcNow -lt $deadline -and -not $ready) {
        if ($api.HasExited) { throw "API exited with code $($api.ExitCode). See $apiLog" }
        $listener = Get-NetTCPConnection -LocalPort 8011 -State Listen -ErrorAction SilentlyContinue
        if ($listener) {
            if (@($listener).Count -ne 1 -or $listener.LocalAddress -ne '127.0.0.1' -or
                -not (Test-OwnedListener $api.Id $listener.OwningProcess)) {
                throw 'Unexpected listener on port 8011.'
            }
            try {
                $probe = Invoke-RestMethod -Uri "$origin/livez" -TimeoutSec 2
                $ready = $probe.service -eq 'slopewatch-public' -and $probe.status -eq 'ok'
            } catch { }
        }
        if (-not $ready) { Start-Sleep -Milliseconds 250 }
    }
    if (-not $ready) { throw "Timed out waiting for the public API. See $apiLog" }
    Write-Host "Public-mode API is ready at $origin. Press Ctrl+C to stop."

    if (-not $LocalOnly) {
        $tunnelLog = Join-Path $logs 'tunnel.log'
        $tunnel = Start-Process -FilePath $connector -ArgumentList 'tunnel','--protocol',$Protocol,'--url',$origin -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardError $tunnelLog -PassThru
        $deadline = [datetime]::UtcNow.AddSeconds(90)
        $publicUrl = $null
    }
    while (-not $api.HasExited) {
        if ($tunnel) {
            if ($tunnel.HasExited) { throw "Cloudflare connector exited with code $($tunnel.ExitCode). See $tunnelLog" }
            if (-not $publicUrl) {
                $output = Get-Content -LiteralPath $tunnelLog -Raw -ErrorAction SilentlyContinue
                if ($output -match 'https://[a-z0-9-]+\.trycloudflare\.com\b') {
                    $publicUrl = $Matches[0]
                    Write-Host "Demo URL: $publicUrl"
                } elseif ([datetime]::UtcNow -gt $deadline) {
                    throw "Cloudflare has not returned a demo URL. See $tunnelLog"
                }
            }
        }
        Start-Sleep -Milliseconds 500
    }
    throw "API exited with code $($api.ExitCode). See $apiLog"
} finally {
    Stop-OwnedTree $tunnel
    Stop-OwnedTree $api
    $env:SLOPEWATCH_API_TOKEN = $originalToken
}
