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
from dotenv import load_dotenv
import logging
import psutil
import gc

# Load environment variables
load_dotenv()

# Configure PyTorch for Cloud Run environment
os.environ['NNPACK_DISABLE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
})

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

@app.route('/api/upload', methods=['POST'])
def upload_file():
    try:
        print("📤 Upload request received")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400
            
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        print(f"📁 Processing file: {file.filename}")
        
        # Read file directly into memory
        file_data = file.read()
        file_size = len(file_data)
        print(f"📊 File size: {file_size / (1024*1024):.2f} MB")
        
        # Check file size limit - reduce for videos
        max_size = 50 * 1024 * 1024 if file.filename.lower().endswith('.mp4') else 100 * 1024 * 1024
        if file_size > max_size:
            return jsonify({"error": f"File too large. Maximum size is {max_size/(1024*1024):.0f}MB for videos."}), 400
        
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # For videos, add early validation
        if file_ext == '.mp4':
            # Quick video duration check using OpenCV
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name
            
            try:
                cap = cv2.VideoCapture(temp_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    # Limit video duration to 30 seconds for Cloud Run
                    if duration > 30:
                        print(f"❌ Video too long: {duration}s > 30s")
                        if temp_path and os.path.exists(temp_path):
                            os.unlink(temp_path)
                        return jsonify({"error": "Video too long. Maximum duration is 30 seconds."}), 400
                    
                    print(f"📹 Video duration: {duration:.1f}s")
                else:
                    os.unlink(temp_path)
                    return jsonify({"error": "Invalid video file"}), 400
                    
                os.unlink(temp_path)
            except Exception as e:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                return jsonify({"error": f"Video validation failed: {str(e)}"}), 400
        
        # Process with YOLO in memory
        print("🤖 Starting YOLO processing...")
        print("⚠️ Note: Video processing may take several minutes, please wait...")
        
        original_base64, processed_base64, mime_type = process_file_in_memory(
            file_data, file_ext, file.filename, model
        )
        
        # Force proper MIME type for web compatibility
        if file_ext.lower() in ['.mp4', '.avi', '.mov', '.webm']:
            mime_type = "video/mp4; codecs='avc1.42E01E'"  # Specific H.264 codec info
        
        print("✅ YOLO processing complete")
        
        # Prepare response
        response_data = {
            "message": "File processed successfully",
            "original": f"data:{mime_type};base64,{original_base64}",
            "processed": f"data:{mime_type};base64,{processed_base64}",
        }
        
        print("📤 Sending response to client...")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in upload_file: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/api/youtube', methods=['POST'])
def process_youtube():
    try:
        data = request.get_json()
        youtube_url = data.get('url')
        
        if not youtube_url:
            return jsonify({"error": "No YouTube URL provided"}), 400
        
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
    # Get port from environment (Google Cloud Run sets PORT automatically)
    port = int(os.environ.get('PORT', 8080))  # Changed default to 8080
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    
    print("🚀 Starting Flask server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔧 Debug mode: {app.config['DEBUG']}")
    print(f"📁 Model path: {model_path}")
    print(f"📏 Max file size: {100 * 1024 * 1024 / (1024*1024):.0f}MB")
    print("💾 Processing files in memory only - no disk storage!")
    print("🔗 YouTube video processing enabled!")
    
    app.run(
        host=host,
        port=port,
        debug=app.config['DEBUG'],
        use_reloader=False,
        threaded=True
    )
