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

# Simple CORS - keep it basic
CORS(app)

# Configure Flask for large files
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['DEBUG'] = False
app.config['TESTING'] = False

# Increase timeout for large file processing
app.config['PERMANENT_SESSION_LIFETIME'] = 1800  # 30 minutes

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
def upload_file():
    try:
        print("🎯 Upload endpoint reached successfully")
        
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Read file data
        file_data = file.read()
        file_size_mb = len(file_data) / (1024 * 1024)
        
        # Simple file size check
        if file_size_mb > 75:
            return jsonify({"error": f"File too large. Maximum size is 75MB."}), 400
        
        print(f"📁 File uploaded: {file.filename} ({file_size_mb:.2f}MB)")
        
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        
        # Simple file type validation
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.webm'}
        if file_ext not in allowed_extensions:
            return jsonify({"error": f"Unsupported file type: {file_ext}"}), 400
        
        # Process with YOLO
        print("🤖 Starting YOLO processing...")
        
        original_base64, processed_base64, mime_type = process_file_in_memory(
            file_data, file_ext, file.filename, model
        )
        
        print("✅ YOLO processing complete")
        
        # Simple response
        response_data = {
            "message": "File processed successfully",
            "original": f"data:{mime_type};base64,{original_base64}",
            "processed": f"data:{mime_type};base64,{processed_base64}",
            "file_size_mb": round(file_size_mb, 2)
        }
        
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
        
        original_base64, processed_base64 = process_youtube_video(youtube_url, model)
        
        print("✅ YouTube video processing complete")
        
        # Prepare response
        response_data = {
            "original_file": f"data:video/webm;base64,{original_base64}",
            "output_file": f"data:video/webm;base64,{processed_base64}",
            "message": "YouTube video processed successfully"
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in process_youtube: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"YouTube processing failed: {str(e)}"}), 500

# Add chunked upload endpoint
@app.route('/api/upload-chunk', methods=['POST'])
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
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/process-uploaded', methods=['POST'])
def process_uploaded():
    upload_id = None
    temp_dir = None
    
    try:
        print("🔄 Starting chunked file processing...")
        
        # Parse request data with better error handling
        try:
            data = request.get_json()
            if not data:
                print("❌ No JSON data received")
                return jsonify({"error": "No JSON data received"}), 400
                
            upload_id = data.get('uploadId')
            file_name = data.get('fileName')
            
            if not upload_id or not file_name:
                print(f"❌ Missing required fields: uploadId={upload_id}, fileName={file_name}")
                return jsonify({"error": "Missing uploadId or fileName"}), 400
                
        except Exception as parse_error:
            print(f"❌ Error parsing request: {str(parse_error)}")
            return jsonify({"error": f"Invalid request format: {str(parse_error)}"}), 400
        
        print(f"📋 Processing upload ID: {upload_id}")
        print(f"📁 File name: {file_name}")
        
        # Check temp directory
        temp_dir = os.path.join(tempfile.gettempdir(), f"upload_{upload_id}")
        
        if not os.path.exists(temp_dir):
            print(f"❌ Temp directory not found: {temp_dir}")
            return jsonify({"error": "Upload chunks not found. Please re-upload the file."}), 400
        
        print(f"📂 Found temp directory: {temp_dir}")
        
        # List all files in temp directory for debugging
        try:
            files_in_dir = os.listdir(temp_dir)
            print(f"📋 Files in temp dir: {files_in_dir}")
        except Exception as list_error:
            print(f"❌ Error listing temp directory: {str(list_error)}")
            return jsonify({"error": "Cannot access upload chunks"}), 500
        
        # Reconstruct file from chunks
        combined_data = bytearray()
        chunk_index = 0
        chunks_found = 0
        
        while True:
            chunk_path = os.path.join(temp_dir, f"chunk_{chunk_index}")
            if not os.path.exists(chunk_path):
                print(f"📋 No more chunks found at index {chunk_index}")
                break
            try:
                with open(chunk_path, 'rb') as f:
                    chunk_data = f.read()
                    combined_data.extend(chunk_data)
                    chunks_found += 1
                    print(f"📦 Loaded chunk {chunk_index}: {len(chunk_data)} bytes")
            except Exception as chunk_error:
                print(f"❌ Error reading chunk {chunk_index}: {str(chunk_error)}")
                return jsonify({"error": f"Error reading chunk {chunk_index}: {str(chunk_error)}"}), 500
            chunk_index += 1
        
        total_size_mb = len(combined_data) / (1024 * 1024)
        print(f"📋 Reconstructed file: {len(combined_data)} bytes ({total_size_mb:.2f}MB) from {chunks_found} chunks")
        
        if len(combined_data) == 0:
            print("❌ No data found in uploaded chunks")
            return jsonify({"error": "No data found in uploaded chunks"}), 400
        
        if chunks_found == 0:
            print("❌ No chunks found")
            return jsonify({"error": "No chunks found for processing"}), 400
        
        # Check if file is too large for processing
        if total_size_mb > 50:  # Conservative limit
            print(f"⚠️ Very large file detected ({total_size_mb:.2f}MB)")
            return jsonify({
                "error": "File too large for processing. Please try with a video under 50MB.",
                "file_size_mb": round(total_size_mb, 2)
            }), 413
        
        # Get file extension for processing
        file_ext = os.path.splitext(file_name)[1].lower()
        print(f"🔧 File extension: {file_ext}")
        
        if file_ext not in ['.mp4', '.avi', '.mov', '.webm']:
            print(f"❌ Unsupported file type for chunked upload: {file_ext}")
            return jsonify({"error": f"Unsupported file type: {file_ext}"}), 400
        
        # Process with YOLO - with enhanced error handling
        print(f"🤖 Processing {file_ext} file with YOLO...")
        
        try:
            original_base64, processed_base64, mime_type = process_file_in_memory(
                bytes(combined_data), file_ext, file_name, model
            )
            print("✅ YOLO processing completed successfully")
            
        except Exception as processing_error:
            print(f"❌ YOLO processing failed: {str(processing_error)}")
            traceback.print_exc()
            
            # Cleanup temp files before returning error
            try:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print("🧹 Cleaned up temp files after processing error")
            except:
                pass
            
            return jsonify({
                "error": f"Video processing failed: {str(processing_error)}",
                "suggestion": "This may be due to file format or size issues. Try with a smaller MP4 file."
            }), 500
        
        # Prepare response with size check
        response_data = {
            "message": "Large file processed successfully",
            "original": f"data:{mime_type};base64,{original_base64}",
            "processed": f"data:{mime_type};base64,{processed_base64}",
            "file_size_mb": round(total_size_mb, 2)
        }
        
        # Check response size
        response_size_mb = len(str(response_data)) / (1024 * 1024)
        print(f"📊 Response size: {response_size_mb:.2f}MB")
        
        # Cleanup temp files
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up temp directory: {temp_dir}")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup warning: {str(cleanup_error)}")
        
        print("✅ Chunked upload processing complete")
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Unexpected error in process_uploaded: {str(e)}")
        traceback.print_exc()
        
        # Emergency cleanup
        try:
            if upload_id:
                emergency_temp_dir = os.path.join(tempfile.gettempdir(), f"upload_{upload_id}")
                if os.path.exists(emergency_temp_dir):
                    shutil.rmtree(emergency_temp_dir)
                    print("🧹 Emergency cleanup completed")
        except:
            pass
        
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

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
    print("📦 Chunked upload support enabled!")
    
    app.run(
        host=host,
        port=port,
        debug=app.config['DEBUG'],
        use_reloader=False,
        threaded=True
    )
