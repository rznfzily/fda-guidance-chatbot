Write-Host "Starting FDA Guidance Chatbot API Server..." -ForegroundColor Green
Write-Host ""

# Run from an activated Python environment (for example: conda activate chatbot)
uvicorn fda_api_main:app --reload --host 0.0.0.0 --port 8000
