# publish.ps1 — Create GitHub repo and push
# Requires: gh CLI  (winget install GitHub.cli)
# Run: .\publish.ps1 -RepoName "kd-distill" -Private

param(
    [string]$RepoName = "kd-distill",
    [switch]$Private,
    [string]$Description = "API/Ollama Teacher → Local GPU Student via Hinton Knowledge Distillation"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Check gh installed
if (-not (Get-Command gh -ErrorAction SilentlyContinue)) {
    Write-Error "gh CLI not found. Install: winget install GitHub.cli  then: gh auth login"
}

# Init git if needed
if (-not (Test-Path ".git")) {
    git init
    git branch -M main
}

git add -A
git commit -m "init: kd-distill pipeline" --allow-empty

# Create GitHub repo
$visibility = if ($Private) { "--private" } else { "--public" }
gh repo create $RepoName $visibility --description $Description --source . --remote origin --push

Write-Host "`n✅ Repo live at: https://github.com/$(gh api user --jq .login)/$RepoName" -ForegroundColor Green
