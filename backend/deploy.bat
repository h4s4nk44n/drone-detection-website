@echo off
echo 🚀 Deploying to Google Cloud Run...
echo.

REM Check if gcloud is available
gcloud version >nul 2>&1
if %errorlevel% neq 0 (
    echo ❌ Google Cloud SDK not found!
    echo.
    echo Please install Google Cloud SDK from:
    echo https://cloud.google.com/sdk/docs/install
    echo.
    echo Or use Google Cloud Console to deploy manually.
    pause
    exit /b 1
)

echo ✅ Google Cloud SDK found
echo.

echo 📦 Building and deploying...
gcloud run deploy drone-detection ^
    --source . ^
    --region europe-west1 ^
    --allow-unauthenticated ^
    --memory 4Gi ^
    --cpu 2 ^
    --timeout 3600 ^
    --max-instances 10

if %errorlevel% equ 0 (
    echo.
    echo ✅ Deployment successful!
    echo 🌐 Your backend is now updated with CORS fixes
) else (
    echo.
    echo ❌ Deployment failed!
    echo Check the error messages above
)

pause 