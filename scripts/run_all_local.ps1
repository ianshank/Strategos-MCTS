param (
    [switch]$SkipServers = $false
)

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Strategos-MCTS Local Validation Suite" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# Set environment variables required for running locally on Windows
$env:PYTHONPATH="."
$env:PYTHONIOENCODING="utf-8"
$env:CUDA_VISIBLE_DEVICES="-1"

Write-Host "`n[1/5] Ensuring python-chess is installed (with distutils workaround)..." -ForegroundColor Yellow
$env:SETUPTOOLS_USE_DISTUTILS="stdlib"
pip install chess --quiet

Write-Host "`n[2/5] Running Zero-Dependency Demos (Tier 1)..." -ForegroundColor Yellow
$demos_tier1 = @(
    "demo.py",
    "chess_demo.py",
    "healthcheck.py",
    "examples/mcp_usage_example.py"
)

foreach ($script in $demos_tier1) {
    Write-Host " -> Running $script" -ForegroundColor Green
    python $script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error running $script" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "`n[3/5] Running GPU-Compatible/Neural Demos on CPU (Tier 2/4)..." -ForegroundColor Yellow
$demos_tier2 = @(
    "src/training/train_rnn.py",
    "src/training/train_bert_lora.py",
    "examples/mcts_determinism_demo.py",
    "examples/neural_training_demo.py",
    "examples/deepmind_style_training.py",
    "examples/advanced_mcts_demo.py",
    "examples/hybrid_agent_demo.py",
    "examples/chess_alphazero_training.py"
)

foreach ($script in $demos_tier2) {
    Write-Host " -> Running $script" -ForegroundColor Green
    python $script
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Error running $script" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

Write-Host "`n[4/5] Running Benchmark and Evaluation Checks (Tier 5)..." -ForegroundColor Yellow
Write-Host " -> Running benchmark dry-run" -ForegroundColor Green
python -m src.benchmark --dry-run
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host " -> Running harness validate-spec" -ForegroundColor Green
python -m src.framework.harness.cli validate-spec specs/phase_0_baseline.SPEC.md
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipServers) {
    Write-Host "`n[5/5] Checking Servers (Tier 3)..." -ForegroundColor Yellow
    Write-Host "Note: Servers run indefinitely. Press Ctrl+C to terminate them." -ForegroundColor Gray
    
    Write-Host " -> To run REST API: python src/api/rest_server.py"
    Write-Host " -> To run Gradio UI: python app.py"
}

Write-Host "`n============================================================" -ForegroundColor Cyan
Write-Host "  All local validations completed successfully!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
