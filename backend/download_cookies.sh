#!/bin/bash
echo "Downloading cookies.txt from GCS..."
curl -o /app/cookies.txt https://storage.googleapis.com/drone-detection-secrets/cookies.txt
