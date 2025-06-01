from flask import Flask, request, jsonify, make_response
import os
import base64
from ultralytics import YOLO
from yolo_utils import process_file_in_memory, process_youtube_video
from flask_cors import CORS, cross_origin
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
import shutil

# Load environment variables
load_dotenv()

# Configure PyTorch for Cloud Run environment
os.environ['NNPACK_DISABLE'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

app = Flask(__name__)

# Simple and comprehensive CORS configuration
CORS(app, 
     origins=["*"],
     methods=["GET", "POST", "OPTIONS"],
     allow_headers=["Content-Type", "Authorization"],
     supports_credentials=False,
     max_age=3600)

# Configure Flask for large files and longer timeouts
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['DEBUG'] = False
app.config['TESTING'] = False
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Increase timeout for large file processing
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

# Add global CORS headers for all responses
@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    response.headers.add('Access-Control-Allow-Credentials', 'false')
    return response

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

@app.before_request
def log_request_info():
    if request.method == 'POST':
        content_length = request.headers.get('Content-Length')
        print(f"📊 Incoming request size: {content_length} bytes")
        if content_length and int(content_length) > 50 * 1024 * 1024:
            print(f"🚨 Large request detected: {int(content_length) / (1024*1024):.1f}MB")

@app.route('/api/upload', methods=['POST'])
@cross_origin()
def upload_file():
    try:
        print("🎯 Upload endpoint reached successfully")
        
        if 'file' not in request.files:
            response = jsonify({"error": "No file uploaded"})
            return response, 400
        
        file = request.files['file']
        if file.filename == '':
            response = jsonify({"error": "No file selected"})
            return response, 400
        
        # Read file data in chunks for large files
        file_data = file.read()
        file_size_mb = len(file_data) / (1024 * 1024)
        
        # Updated file size limit: 75MB
        MAX_FILE_SIZE_MB = 75
        if file_size_mb > MAX_FILE_SIZE_MB:
            response = jsonify({"error": f"File too large. Maximum size is {MAX_FILE_SIZE_MB}MB."})
            return response, 400
        
        print(f"📁 File uploaded: {file.filename} ({file_size_mb:.2f}MB)")
        
        # Add processing time estimate
        if file_size_mb > 30:
            print(f"🚨 LARGE FILE ALERT: {file_size_mb:.2f}MB will take 45-90 minutes to process")
            print("🔧 Using maximum resources for processing...")
        elif file_size_mb > 20:
            print(f"⚠️ Large file: {file_size_mb:.2f}MB will take 30-60 minutes to process")
        
        # Force garbage collection before processing
        gc.collect()
        
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # Validate file type
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.webm'}
        if file_ext not in allowed_extensions:
            response = jsonify({"error": f"Unsupported file type: {file_ext}"})
            return response, 400
        
        # For videos, add early validation
        if file_ext in ['.mp4', '.avi', '.mov', '.webm']:
            # Quick video duration check using OpenCV
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_file:
                temp_file.write(file_data)
                temp_path = temp_file.name
            
            try:
                cap = cv2.VideoCapture(temp_path)
                if cap.isOpened():
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                    
                    # Duration limit: 120 seconds
                    if duration > 120:
                        print(f"❌ Video too long: {duration}s > 120s")
                        if temp_path and os.path.exists(temp_path):
                            os.unlink(temp_path)
                        response = jsonify({"error": "Video too long. Maximum duration is 120 seconds."})
                        return response, 400
                    
                    print(f"📹 Video duration: {duration:.1f}s - ✅ Within 120s limit")
                    print(f"📊 File size: {file_size_mb:.2f}MB - ✅ Within 75MB limit")
                    
                    # Estimate processing time for user feedback
                    estimated_time = max(2, int(duration * 0.5))  # Rough estimate
                    print(f"⏱️ Estimated processing time: ~{estimated_time} minutes")
                else:
                    os.unlink(temp_path)
                    response = jsonify({"error": "Invalid video file"})
                    return response, 400
                    
                os.unlink(temp_path)
            except Exception as e:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                response = jsonify({"error": f"Video validation failed: {str(e)}"})
                return response, 400
        
        # Process with YOLO in memory
        print("🤖 Starting YOLO processing...")
        if file_size_mb > 20:
            print("⚠️ Large file detected - processing may take 15-20 minutes, please wait...")
        else:
            print("⚠️ Note: Video processing may take several minutes, please wait...")
        
        original_base64, processed_base64, mime_type = process_file_in_memory(
            file_data, file_ext, file.filename, model
        )
        
        # Force proper MIME type for web compatibility
        if file_ext.lower() in ['.mp4', '.avi', '.mov', '.webm']:
            mime_type = "video/mp4; codecs='avc1.42E01E'"  # Specific H.264 codec info
        
        print("✅ YOLO processing complete")
        
        # Prepare response with explicit CORS headers
        response_data = {
            "message": "File processed successfully",
            "original": f"data:{mime_type};base64,{original_base64}",
            "processed": f"data:{mime_type};base64,{processed_base64}",
            "file_size_mb": round(file_size_mb, 2)
        }
        
        print("📤 Sending response to client...")
        response = jsonify(response_data)
        return response
        
    except Exception as e:
        print(f"❌ Error in upload_file: {str(e)}")
        traceback.print_exc()
        error_response = jsonify({"error": f"Processing failed: {str(e)}"})
        return error_response, 500

@app.route('/api/youtube', methods=['POST'])
@cross_origin()
def process_youtube():
    try:
        data = request.get_json()
        youtube_url = data.get('url')
        
        if not youtube_url:
            response = jsonify({"error": "No YouTube URL provided"})
            return response, 400
        
        print(f"🔗 Processing YouTube URL: {youtube_url}")
        
        # Validate YouTube URL
        if not is_youtube_url(youtube_url):
            response = jsonify({"error": "Invalid YouTube URL format"})
            return response, 400
        
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
        response = jsonify(response_data)
        return response
        
    except Exception as e:
        print(f"❌ Error in process_youtube: {str(e)}")
        traceback.print_exc()
        error_response = jsonify({"error": f"YouTube processing failed: {str(e)}"})
        return error_response, 500

# Add chunked upload endpoint
@app.route('/api/upload-chunk', methods=['POST'])
@cross_origin()
def upload_chunk():
    try:
        chunk = request.files['chunk']
        chunk_index = int(request.form['chunkIndex'])
        total_chunks = int(request.form['totalChunks'])
        upload_id = request.form['uploadId']
        file_name = request.form['fileName']
        
        # Create temp directory for this upload
        temp_dir = os.path.join(tempfile.gettempdir(), f"upload_{upload_id}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save chunk
        chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index}")
        chunk.save(chunk_path)
        
        print(f"📦 Saved chunk {chunk_index + 1}/{total_chunks}")
        
        return jsonify({"message": f"Chunk {chunk_index + 1} uploaded successfully"})
        
    except Exception as e:
        print(f"❌ Chunk upload error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-uploaded', methods=['POST', 'OPTIONS'])
@cross_origin()
def process_uploaded():
    if request.method == 'OPTIONS':
        response = make_response()
        return response
        
    try:
        # Clear memory before processing
        gc.collect()
        
        # Check available memory
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        print(f"💾 Available memory: {available_memory:.2f}MB")
        
        # Reduce memory threshold for Cloud Run (2GB instances)
        if available_memory < 500:  # Less than 500MB available
            return jsonify({"error": "Insufficient memory available for processing"}), 503
        
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400
            
        upload_id = data.get('uploadId')
        file_name = data.get('fileName')
        
        if not upload_id or not file_name:
            return jsonify({"error": "Missing uploadId or fileName"}), 400
        
        # Reconstruct file from chunks
        temp_dir = os.path.join(tempfile.gettempdir(), f"upload_{upload_id}")
        if not os.path.exists(temp_dir):
            return jsonify({"error": "Upload directory not found"}), 404
        
        try:
            # Combine chunks with progress tracking
            combined_data = bytearray()
            chunk_index = 0
            total_size = 0
            
            print(f"🔄 Reconstructing file from chunks in {temp_dir}")
            
            while True:
                chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index}")
                if not os.path.exists(chunk_path):
                    break
                    
                with open(chunk_path, 'rb') as f:
                    chunk_data = f.read()
                    combined_data.extend(chunk_data)
                    total_size += len(chunk_data)
                    print(f"📦 Added chunk {chunk_index}, total size: {total_size / (1024*1024):.2f}MB")
                
                chunk_index += 1
            
            if total_size == 0:
                raise Exception("No chunks found")
                
            print(f"📋 Reconstructed file: {total_size / (1024*1024):.2f}MB")
            
            # Process with YOLO
            file_ext = os.path.splitext(file_name)[1].lower()
            
            # Memory check before processing
            available_memory = psutil.virtual_memory().available / (1024 * 1024)
            print(f"💾 Available memory before processing: {available_memory:.2f}MB")
            
            # More conservative memory check for large files
            if total_size > 50 * 1024 * 1024 and available_memory < 1000:  # 50MB file needs 1GB free
                raise Exception("Insufficient memory available for large file processing")
            elif total_size > 20 * 1024 * 1024 and available_memory < 500:  # 20MB file needs 500MB free
                raise Exception("Insufficient memory available for medium file processing")
            
            original_base64, processed_base64, mime_type = process_file_in_memory(
                bytes(combined_data), file_ext, file_name, model
            )
            
            # Cleanup temp files immediately
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            
            # Force garbage collection after processing
            gc.collect()
            
            response = jsonify({
                "message": "Large file processed successfully",
                "original": f"data:{mime_type};base64,{original_base64}",
                "processed": f"data:{mime_type};base64,{processed_base64}"
            })
            
            return response
            
        except Exception as process_error:
            # Cleanup on error
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            raise process_error
            
    except Exception as e:
        print(f"❌ Process uploaded error: {str(e)}")
        traceback.print_exc()
        
        error_response = jsonify({
            "error": str(e),
            "details": traceback.format_exc()
        })
        return error_response, 500

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
