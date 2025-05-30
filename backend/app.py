from flask import Flask, request, jsonify
import os
import base64
from ultralytics import YOLO
from yolo_utils import process_file_in_memory, process_youtube_video
from flask_cors import CORS
import traceback
import tempfile
import cv2
import numpy as np
from io import BytesIO
import yt_dlp
import re

app = Flask(__name__)
CORS(app)

# Disable all debug features
app.config['DEBUG'] = False
app.config['TESTING'] = False

def is_youtube_url(url):
    """
    Check if the URL is a valid YouTube URL
    """
    youtube_patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/[\w-]+',
        r'(?:https?://)?youtu\.be/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/[\w-]+',
        r'(?:https?://)?(?:www\.)?youtube\.com/v/[\w-]+',
        r'(?:https?://)?(?:m\.)?youtube\.com/watch\?v=[\w-]+',
    ]
    
    for pattern in youtube_patterns:
        if re.match(pattern, url):
            return True
    return False

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response

@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Flask server is running on port 5001!"})

@app.route('/test', methods=['GET'])
def test():
    return jsonify({"message": "Test endpoint working!", "status": "ok"})

# Load model
print("🤖 Loading YOLO model...")
model_path = os.path.join(os.path.dirname(__file__), 'models', 'best.pt')
model = YOLO(model_path)
print("✅ YOLO model loaded successfully")

@app.route('/api/upload', methods=['POST', 'OPTIONS'])
def upload_file():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'}), 200
        
    try:
        print("📤 Upload request received")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        print(f"📁 Processing file: {file.filename}")
        
        # Read file directly into memory
        file_data = file.read()
        file_size = len(file_data)
        print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
        
        # Check file size limit
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            return jsonify({"error": "File too large. Maximum size is 100MB."}), 400
        
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # Process with YOLO in memory
        print("🤖 Starting YOLO processing...")
        print("⚠️ Note: Video processing may take several minutes, please wait...")
        
        original_base64, processed_base64, mime_type = process_file_in_memory(
            file_data, file_ext, file.filename, model
        )
        
        print("✅ YOLO processing complete")
        
        # Prepare response
        response_data = {
            "original_file": f"data:{mime_type};base64,{original_base64}",
            "output_file": f"data:{mime_type};base64,{processed_base64}",
            "message": "File processed successfully"
        }
        
        print("📤 Sending response to client...")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in upload_file: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/api/youtube', methods=['POST', 'OPTIONS'])
def process_youtube():
    if request.method == 'OPTIONS':
        return jsonify({'status': 'OK'}), 200
        
    try:
        print("🔗 YouTube processing request received")
        
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({"error": "No YouTube URL provided"}), 400
        
        youtube_url = data['url'].strip()
        print(f"🔗 Processing YouTube URL: {youtube_url}")
        
        # Validate YouTube URL
        if not is_youtube_url(youtube_url):
            return jsonify({"error": "Invalid YouTube URL format"}), 400
        
        # Process YouTube video
        print("🤖 Starting YouTube video processing...")
        print("⚠️ Note: Downloading and processing may take several minutes, please wait...")
        
        original_base64, processed_base64 = process_youtube_video(youtube_url, model)
        
        print("✅ YouTube video processing complete")
        
        # Prepare response
        response_data = {
            "original_file": f"data:video/webm;base64,{original_base64}",
            "output_file": f"data:video/webm;base64,{processed_base64}",
            "message": "YouTube video processed successfully"
        }
        
        print("📤 Sending response to client...")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in process_youtube: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"YouTube processing failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("🚀 Starting Flask server in PRODUCTION mode...")
    print("💾 Processing files in memory only - no disk storage!")
    print("🔗 YouTube video processing enabled!")
    print("⚠️ Debug mode is DISABLED to prevent restarts during processing")
    
    app.run(
        host='localhost', 
        port=5001, 
        debug=False,
        use_reloader=False,
        threaded=True
    )
