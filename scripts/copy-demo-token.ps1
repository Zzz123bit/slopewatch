[CmdletBinding()]
param([switch]$InitializeOnly)

$ErrorActionPreference = 'Stop'
$token = $env:SLOPEWATCH_API_TOKEN
if ([string]::IsNullOrEmpty($token)) {
    $directory = Join-Path $env:LOCALAPPDATA 'SlopeWatch'
    [IO.Directory]::CreateDirectory($directory) | Out-Null
    $cache = Join-Path $directory 'api-token.dpapi'
    if (-not [IO.File]::Exists($cache)) {
        $bytes = New-Object byte[] 32
        $rng = [Security.Cryptography.RandomNumberGenerator]::Create()
        try { $rng.GetBytes($bytes) } finally { $rng.Dispose() }
        $generated = [Convert]::ToBase64String($bytes).TrimEnd('=').Replace('+','-').Replace('/','_')
        $secure = ConvertTo-SecureString $generated -AsPlainText -Force
        $temp = Join-Path $directory ([guid]::NewGuid().ToString('N') + '.tmp')
        try {
            [IO.File]::WriteAllText($temp, (ConvertFrom-SecureString $secure), [Text.Encoding]::ASCII)
            try {
                [IO.File]::Move($temp, $cache)
            } catch [IO.IOException] {
                # If another invocation won the race, use the value it saved.
                if (-not [IO.File]::Exists($cache)) { throw }
            }
        } finally {
            $secure.Dispose()
            [IO.File]::Delete($temp)
        }
    }
    $secure = ConvertTo-SecureString ([IO.File]::ReadAllText($cache).Trim())
    try { $token = [Net.NetworkCredential]::new('', $secure).Password } finally { $secure.Dispose() }
}
if ($token -cnotmatch '\A[A-Za-z0-9_-]{32,}\z') {
    throw 'Supply a URL-safe SLOPEWATCH_API_TOKEN of at least 32 characters.'
}
if ($InitializeOnly) {
    Write-Host 'Demo token is ready.'
} else {
    Set-Clipboard -Value $token
    Write-Host 'Demo token copied to the clipboard.'
}
