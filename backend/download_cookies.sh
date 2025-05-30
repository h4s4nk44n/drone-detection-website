#!/bin/bash
echo "Downloading cookies.txt from GCS..."
gsutil cp gs://drone-detection-secrets/cookies.txt /app/cookies.txt
