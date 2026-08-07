[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding            = [System.Text.Encoding]::UTF8
chcp 65001 | Out-Null

$GH_USER = "jinhae8971"
$GH_REPO = "cross-market-flow-radar"

if ($env:GH_TOKEN) { $GH_TOKEN = $env:GH_TOKEN } else {
    $sec = Read-Host "GitHub PAT (repo 스코프)" -AsSecureString
    $GH_TOKEN = [System.Net.NetworkCredential]::new("", $sec).Password
}
if (-not $GH_TOKEN) { Write-Host "토큰 필요" -ForegroundColor Red; exit 1 }

$REMOTE_URL = "https://$GH_TOKEN@github.com/$GH_USER/$GH_REPO.git"
$API_HDR = @{
    "Authorization" = "token $GH_TOKEN"
    "Accept"        = "application/vnd.github+json"
    "User-Agent"    = "FlowRadarDeploy"
}
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

git config --global --add safe.directory ($ScriptDir -replace '\\','/') 2>$null
if (-not (Test-Path ".git")) { git init | Out-Null }
$prev = $ErrorActionPreference; $ErrorActionPreference = "SilentlyContinue"
git remote remove origin 2>$null | Out-Null
$ErrorActionPreference = $prev
git remote add origin $REMOTE_URL
git config user.name $GH_USER; git config user.email "jinhae8971@gmail.com"
Write-Host "[1] Git OK" -ForegroundColor Green

try {
    Invoke-RestMethod -Uri "https://api.github.com/repos/$GH_USER/$GH_REPO" -Headers $API_HDR | Out-Null
    Write-Host "[2] Repo exists" -ForegroundColor Green
} catch {
    try {
        Invoke-RestMethod -Method Post -Uri "https://api.github.com/user/repos" -Headers $API_HDR `
            -Body (@{name=$GH_REPO;private=$false;auto_init=$false} | ConvertTo-Json) `
            -ContentType "application/json" | Out-Null
        Write-Host "[2] Repo created" -ForegroundColor Green; Start-Sleep -Seconds 2
    } catch {
        Write-Host "[2] 직접 생성: https://github.com/new (name: $GH_REPO)" -ForegroundColor Red
        Read-Host "생성 후 Enter"
    }
}

$ErrorActionPreference = "SilentlyContinue"
git add .; git commit -m "feat: cross-market flow radar" 2>$null
if ($LASTEXITCODE -ne 0) { git commit --allow-empty -m "chore: update" 2>$null }
git branch -M main; git push -u origin main --force 2>$null
$pushCode = $LASTEXITCODE; $ErrorActionPreference = "Stop"
if ($pushCode -ne 0) {
    Write-Host "PUSH FAILED. 토큰에 'repo' 스코프 필요" -ForegroundColor Red; exit 1
}
Write-Host "[3] Push OK" -ForegroundColor Green

# [4] GitHub Pages 활성화 (main / docs) - API로 자동 설정
$pagesBody = @{ source = @{ branch = "main"; path = "/docs" } } | ConvertTo-Json -Depth 3
try {
    Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$GH_USER/$GH_REPO/pages" `
        -Headers $API_HDR -Body $pagesBody -ContentType "application/json" | Out-Null
    Write-Host "[4] Pages 생성 (main /docs)" -ForegroundColor Green
} catch {
    try {
        Invoke-RestMethod -Method Put -Uri "https://api.github.com/repos/$GH_USER/$GH_REPO/pages" `
            -Headers $API_HDR -Body $pagesBody -ContentType "application/json" | Out-Null
        Write-Host "[4] Pages 소스 갱신 (main /docs)" -ForegroundColor Green
    } catch {
        Write-Host "[4] Pages 수동 설정 필요: Settings > Pages > main / docs" -ForegroundColor Yellow
    }
}

# [5] DASHBOARD_URL 변수 등록 - 브리프 링크에 쓰인다
$PAGES_URL = "https://$($GH_USER.ToLower()).github.io/$GH_REPO/"
$varBody = @{ name = "DASHBOARD_URL"; value = $PAGES_URL } | ConvertTo-Json
try {
    Invoke-RestMethod -Method Post -Uri "https://api.github.com/repos/$GH_USER/$GH_REPO/actions/variables" `
        -Headers $API_HDR -Body $varBody -ContentType "application/json" | Out-Null
} catch {
    try {
        Invoke-RestMethod -Method Patch `
            -Uri "https://api.github.com/repos/$GH_USER/$GH_REPO/actions/variables/DASHBOARD_URL" `
            -Headers $API_HDR -Body $varBody -ContentType "application/json" | Out-Null
    } catch { Write-Host "[5] DASHBOARD_URL 수동 등록 필요" -ForegroundColor Yellow }
}
Write-Host "[5] DASHBOARD_URL = $PAGES_URL" -ForegroundColor Green

# [6] Secrets 등록 - gh CLI가 있으면 자동, 없으면 안내
$secretNames = @("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "KRX_API_KEY")
if (Get-Command gh -ErrorAction SilentlyContinue) {
    $env:GH_TOKEN = $GH_TOKEN
    foreach ($n in $secretNames) {
        $sv = Read-Host "$n 값 (건너뛰려면 Enter)" -AsSecureString
        $plain = [System.Net.NetworkCredential]::new("", $sv).Password
        if ($plain) { gh secret set $n --body $plain --repo "$GH_USER/$GH_REPO" 2>$null }
    }
    Write-Host "[6] Secrets 등록 완료" -ForegroundColor Green
} else {
    Write-Host "[6] Secrets 등록: https://github.com/$GH_USER/$GH_REPO/settings/secrets/actions" -ForegroundColor Yellow
    Write-Host "    TELEGRAM_TOKEN / TELEGRAM_CHAT_ID / KRX_API_KEY(승인 후)" -ForegroundColor Cyan
    Read-Host "등록 후 Enter"
}

try {
    Invoke-RestMethod -Method Post `
        -Uri "https://api.github.com/repos/$GH_USER/$GH_REPO/actions/workflows/collect.yml/dispatches" `
        -Headers $API_HDR -Body '{"ref":"main","inputs":{"seed":"true"}}' -ContentType "application/json" | Out-Null
    Write-Host "[7] 시딩 실행 트리거 완료 - 약 5분 후 텔레그램 도착" -ForegroundColor Green
} catch {
    Write-Host "[7] 수동 실행: https://github.com/$GH_USER/$GH_REPO/actions" -ForegroundColor White
}
Write-Host "대시보드: $PAGES_URL" -ForegroundColor Cyan

git remote set-url origin "https://github.com/$GH_USER/$GH_REPO.git"
Remove-Variable GH_TOKEN -ErrorAction SilentlyContinue
$env:GH_TOKEN = $null
Write-Host "DONE" -ForegroundColor Cyan
