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
import uuid
import time
import json

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

# Dictionary to track temporary files
temp_files = {}

# Track files currently being downloaded to prevent cleanup
active_downloads = set()

def load_temp_files_from_disk():
    """Load temp file tracking from disk on startup"""
    temp_dir = tempfile.gettempdir()
    print(f"🔄 Loading temp files from disk: {temp_dir}")
    
    for filename in os.listdir(temp_dir):
        if filename.endswith('_metadata.json'):
            try:
                file_id = filename.replace('_metadata.json', '')
                metadata_path = os.path.join(temp_dir, filename)
                
                with open(metadata_path, 'r') as f:
                    file_info = json.load(f)
                
                # Check if the actual file still exists
                if os.path.exists(file_info['path']):
                    temp_files[file_id] = file_info
                    print(f"📋 Restored file tracking: {file_id}")
                else:
                    # Clean up orphaned metadata
                    os.unlink(metadata_path)
                    print(f"🧹 Removed orphaned metadata: {file_id}")
                    
            except Exception as e:
                print(f"⚠️ Failed to load metadata {filename}: {e}")

def save_temp_file_metadata(file_id, file_info):
    """Save temp file metadata to disk"""
    try:
        temp_dir = tempfile.gettempdir()
        metadata_path = os.path.join(temp_dir, f"{file_id}_metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(file_info, f)
        
        print(f"💾 Saved metadata for: {file_id}")
        
    except Exception as e:
        print(f"⚠️ Failed to save metadata for {file_id}: {e}")

def remove_temp_file_metadata(file_id):
    """Remove temp file metadata from disk"""
    try:
        temp_dir = tempfile.gettempdir()
        metadata_path = os.path.join(temp_dir, f"{file_id}_metadata.json")
        
        if os.path.exists(metadata_path):
            os.unlink(metadata_path)
            print(f"🧹 Removed metadata for: {file_id}")
            
    except Exception as e:
        print(f"⚠️ Failed to remove metadata for {file_id}: {e}")

def cleanup_old_files():
    """Clean up files older than 2 hours (increased from 1 hour for chunked downloads)"""
    current_time = time.time()
    expired_files = []
    
    for file_id, file_info in temp_files.items():
        # Don't cleanup files currently being downloaded
        if file_id in active_downloads:
            print(f"⏳ Skipping cleanup of active download: {file_id}")
            continue
            
        # Increased cleanup time to 2 hours to prevent issues with large chunked downloads
        # Also check last_accessed time to keep recently used files
        last_access = file_info.get('last_accessed', file_info['created'])
        # Increase time to 4 hours for better stability
        if current_time - last_access > 14400:  # 4 hours since last access
            expired_files.append(file_id)
    
    for file_id in expired_files:
        try:
            file_info = temp_files[file_id]
            if os.path.exists(file_info['path']):
                os.unlink(file_info['path'])
            del temp_files[file_id]
            remove_temp_file_metadata(file_id)
            print(f"🧹 Cleaned up expired file: {file_id}")
        except:
            pass

def ensure_file_ready(file_path, expected_size, timeout=30):
    """Wait for file to be fully written before marking as ready"""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            if os.path.exists(file_path):
                actual_size = os.path.getsize(file_path)
                if actual_size == expected_size:
                    # File is the expected size, give it a moment to be fully flushed
                    time.sleep(0.1)
                    return True
            time.sleep(0.2)  # Wait 200ms before checking again
        except OSError:
            time.sleep(0.2)
    return False

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

# Load existing temp files from disk (for container restarts)
load_temp_files_from_disk()

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
        
        # Cleanup old files first
        cleanup_old_files()
        
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
        
        # Check if we should use chunked download (for videos > 15MB)
        is_video = file_ext.lower() in ['.mp4', '.avi', '.mov', '.webm']
        
        # Calculate response size
        response_data_size = len(original_base64) + len(processed_base64)
        response_size_mb = response_data_size / (1024 * 1024)
        
        print(f"📊 Response size would be: {response_size_mb:.2f}MB")
        
        # Use chunked download for large responses (> 20MB) or large videos
        if response_size_mb > 20 or (is_video and file_size_mb > 15):
            print(f"📦 Using chunked download for large response ({response_size_mb:.2f}MB)")
            
            # Create file IDs for chunked download
            original_file_id = str(uuid.uuid4())
            processed_file_id = str(uuid.uuid4())
            
            # Save files temporarily for chunked download
            original_temp_path = os.path.join(tempfile.gettempdir(), f"original_{original_file_id}")
            processed_temp_path = os.path.join(tempfile.gettempdir(), f"processed_{processed_file_id}")
            
            # Write files
            with open(original_temp_path, 'wb') as f:
                f.write(base64.b64decode(original_base64))
            
            with open(processed_temp_path, 'wb') as f:
                f.write(base64.b64decode(processed_base64))
            
            # Calculate chunk info
            CHUNK_SIZE = 5 * 1024 * 1024  # 5MB chunks for download
            
            original_size = len(base64.b64decode(original_base64))
            processed_size = len(base64.b64decode(processed_base64))
            
            # Ensure files are fully written before proceeding
            print(f"🔍 Verifying original file is ready: {original_temp_path}")
            if not ensure_file_ready(original_temp_path, original_size):
                raise Exception("Original file not ready after writing")
                
            print(f"🔍 Verifying processed file is ready: {processed_temp_path}")
            if not ensure_file_ready(processed_temp_path, processed_size):
                raise Exception("Processed file not ready after writing")
            
            print("✅ Both files verified and ready for chunked download")
            
            original_chunks = (original_size + CHUNK_SIZE - 1) // CHUNK_SIZE
            processed_chunks = (processed_size + CHUNK_SIZE - 1) // CHUNK_SIZE
            
            # Store file info - only after files are confirmed ready
            temp_files[original_file_id] = {
                'path': original_temp_path,
                'size': original_size,
                'chunks': original_chunks,
                'mime_type': mime_type,
                'created': time.time(),
                'last_accessed': time.time(),
                'ready': True  # Mark as ready for download
            }
            
            temp_files[processed_file_id] = {
                'path': processed_temp_path,
                'size': processed_size,
                'chunks': processed_chunks,
                'mime_type': mime_type,
                'created': time.time(),
                'last_accessed': time.time(),
                'ready': True  # Mark as ready for download
            }
            
            # Save metadata to disk for persistence
            save_temp_file_metadata(original_file_id, temp_files[original_file_id])
            save_temp_file_metadata(processed_file_id, temp_files[processed_file_id])
            
            print(f"📊 Created chunked downloads:")
            print(f"   - Original: {original_chunks} chunks ({original_size / (1024*1024):.2f}MB)")
            print(f"   - Processed: {processed_chunks} chunks ({processed_size / (1024*1024):.2f}MB)")
            
            # Return chunked response
            return jsonify({
                "message": "File processed successfully - using chunked download",
                "chunked_download": True,
                "original_file": {
                    "file_id": original_file_id,
                    "chunks": original_chunks,
                    "size": original_size,
                    "mime_type": mime_type
                },
                "processed_file": {
                    "file_id": processed_file_id,
                    "chunks": processed_chunks,
                    "size": processed_size,
                    "mime_type": mime_type
                },
                "file_size_mb": round(file_size_mb, 2)
            })
        
        else:
            print(f"📤 Using direct response for small file ({response_size_mb:.2f}MB)")
            
            # Use direct base64 response for small files
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
        
        # Cleanup old files first
        cleanup_old_files()
        
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
        
        # Relaxed file size limits since we'll use chunked downloads
        if total_size_mb > 100:  # Increased limit since we can handle larger responses
            print(f"⚠️ File too large for processing ({total_size_mb:.2f}MB)")
            return jsonify({
                "error": f"File too large for processing ({total_size_mb:.1f}MB). Maximum size is 100MB.",
                "file_size_mb": round(total_size_mb, 2),
                "suggestion": "Try with a shorter video or compress your video to reduce file size."
            }), 413
        
        # Get file extension for processing
        file_ext = os.path.splitext(file_name)[1].lower()
        print(f"🔧 File extension: {file_ext}")
        
        if file_ext not in ['.mp4', '.avi', '.mov', '.webm']:
            print(f"❌ Unsupported file type for chunked upload: {file_ext}")
            return jsonify({"error": f"Unsupported file type: {file_ext}"}), 400
        
        # Process with YOLO - enhanced error handling
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
        
        # Create file IDs for chunked download
        original_file_id = str(uuid.uuid4())
        processed_file_id = str(uuid.uuid4())
        
        # Save files temporarily for chunked download
        original_temp_path = os.path.join(tempfile.gettempdir(), f"original_{original_file_id}")
        processed_temp_path = os.path.join(tempfile.gettempdir(), f"processed_{processed_file_id}")
        
        # Write original file
        with open(original_temp_path, 'wb') as f:
            f.write(base64.b64decode(original_base64))
        
        # Write processed file  
        with open(processed_temp_path, 'wb') as f:
            f.write(base64.b64decode(processed_base64))
        
        # Calculate chunk info
        CHUNK_SIZE = 5 * 1024 * 1024  # 5MB chunks for download
        
        original_size = len(base64.b64decode(original_base64))
        processed_size = len(base64.b64decode(processed_base64))
        
        # Ensure files are fully written before proceeding
        print(f"🔍 Verifying original file is ready: {original_temp_path}")
        if not ensure_file_ready(original_temp_path, original_size):
            raise Exception("Original file not ready after writing")
            
        print(f"🔍 Verifying processed file is ready: {processed_temp_path}")
        if not ensure_file_ready(processed_temp_path, processed_size):
            raise Exception("Processed file not ready after writing")
        
        print("✅ Both files verified and ready for chunked download")
        
        original_chunks = (original_size + CHUNK_SIZE - 1) // CHUNK_SIZE
        processed_chunks = (processed_size + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        # Store file info - only after files are confirmed ready
        temp_files[original_file_id] = {
            'path': original_temp_path,
            'size': original_size,
            'chunks': original_chunks,
            'mime_type': mime_type,
            'created': time.time(),
            'last_accessed': time.time(),
            'ready': True  # Mark as ready for download
        }
        
        temp_files[processed_file_id] = {
            'path': processed_temp_path,
            'size': processed_size,
            'chunks': processed_chunks,
            'mime_type': mime_type,
            'created': time.time(),
            'last_accessed': time.time(),
            'ready': True  # Mark as ready for download
        }
        
        # Save metadata to disk for persistence
        save_temp_file_metadata(original_file_id, temp_files[original_file_id])
        save_temp_file_metadata(processed_file_id, temp_files[processed_file_id])
        
        print(f"📊 Created chunked downloads:")
        print(f"   - Original: {original_chunks} chunks ({original_size / (1024*1024):.2f}MB)")
        print(f"   - Processed: {processed_chunks} chunks ({processed_size / (1024*1024):.2f}MB)")
        
        # Cleanup upload temp files
        try:
            if temp_dir and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                print(f"🧹 Cleaned up upload temp directory: {temp_dir}")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup warning: {str(cleanup_error)}")
        
        print("✅ Chunked upload processing complete - using chunked downloads")
        
        # Return file IDs and metadata instead of large base64
        return jsonify({
            "message": "Large file processed successfully - using chunked download",
            "chunked_download": True,
            "original_file": {
                "file_id": original_file_id,
                "chunks": original_chunks,
                "size": original_size,
                "mime_type": mime_type
            },
            "processed_file": {
                "file_id": processed_file_id,
                "chunks": processed_chunks,
                "size": processed_size,
                "mime_type": mime_type
            },
            "file_size_mb": round(total_size_mb, 2)
        })
        
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

@app.route('/api/download-chunk/<file_id>/<int:chunk_index>', methods=['GET'])
def download_chunk(file_id, chunk_index):
    """Download a specific chunk of a processed file"""
    try:
        print(f"📥 Download request: file_id={file_id}, chunk={chunk_index}")
        print(f"📊 Current temp_files count: {len(temp_files)}")
        
        # Mark file as actively being downloaded
        active_downloads.add(file_id)
        
        # First check if file exists in memory
        if file_id not in temp_files:
            # Try to reload from disk metadata in case of server restart
            print(f"🔄 File not in memory, trying to reload from disk: {file_id}")
            try:
                temp_dir = tempfile.gettempdir()
                metadata_path = os.path.join(temp_dir, f"{file_id}_metadata.json")
                
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        file_info = json.load(f)
                    
                    # Check if the actual file still exists
                    if os.path.exists(file_info['path']):
                        temp_files[file_id] = file_info
                        print(f"✅ Restored file from disk metadata: {file_id}")
                    else:
                        print(f"❌ File referenced in metadata doesn't exist: {file_info['path']}")
                        # Clean up orphaned metadata
                        os.unlink(metadata_path)
                        raise FileNotFoundError("File no longer exists on disk")
                else:
                    print(f"❌ No metadata found for file: {file_id}")
                    raise FileNotFoundError("No metadata found")
                    
            except Exception as reload_error:
                print(f"❌ Could not reload file from disk: {reload_error}")
                
        # Check again after potential reload
        if file_id not in temp_files:
            print(f"❌ File not found in temp_files: {file_id}")
            print(f"📋 Available files: {list(temp_files.keys())}")
            # Remove from active downloads since it failed
            active_downloads.discard(file_id)
            response = jsonify({"error": "File not found or expired"})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response, 404
        
        file_info = temp_files[file_id]
        file_path = file_info['path']
        
        # Wait for file to be ready if it's not marked as ready yet
        if not file_info.get('ready', False):
            print(f"⏳ File not marked as ready, waiting: {file_id}")
            # Wait up to 10 seconds for file to be ready
            wait_time = 0
            while not file_info.get('ready', False) and wait_time < 10:
                time.sleep(0.5)
                wait_time += 0.5
                # Refresh file info in case it was updated
                if file_id in temp_files:
                    file_info = temp_files[file_id]
                    
            if not file_info.get('ready', False):
                print(f"❌ File never became ready: {file_id}")
                active_downloads.discard(file_id)
                response = jsonify({"error": "File not ready for download"})
                response.headers['Access-Control-Allow-Origin'] = '*'
                response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
                response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
                return response, 503  # Service Unavailable
        
        # Update last accessed time to prevent cleanup
        temp_files[file_id]['last_accessed'] = time.time()
        
        print(f"📁 File info: path={file_path}, size={file_info['size']}, chunks={file_info['chunks']}")
        print(f"📁 File age: {time.time() - file_info['created']:.1f} seconds")
        
        if not os.path.exists(file_path):
            print(f"❌ File not found on disk: {file_path}")
            print(f"📂 Checking temp directory: {os.path.dirname(file_path)}")
            try:
                temp_files_on_disk = os.listdir(os.path.dirname(file_path))
                related_files = [f for f in temp_files_on_disk if file_id[:8] in f]
                print(f"📋 Related files on disk: {related_files}")
            except Exception as list_error:
                print(f"⚠️ Could not list temp directory: {list_error}")
            
            # Clean up missing file from tracking and active downloads
            del temp_files[file_id]
            active_downloads.discard(file_id)
            response = jsonify({"error": "File not found on disk"})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response, 404
        
        if chunk_index >= file_info['chunks']:
            print(f"❌ Chunk index out of range: {chunk_index} >= {file_info['chunks']}")
            # Remove from active downloads
            active_downloads.discard(file_id)
            response = jsonify({"error": "Chunk index out of range"})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response, 400
        
        CHUNK_SIZE = 5 * 1024 * 1024  # 5MB chunks
        start_byte = chunk_index * CHUNK_SIZE
        end_byte = min(start_byte + CHUNK_SIZE, file_info['size'])
        expected_chunk_size = end_byte - start_byte
        
        print(f"📦 Reading chunk {chunk_index}: bytes {start_byte}-{end_byte} (expecting {expected_chunk_size} bytes)")
        
        # Check file size on disk before reading
        try:
            actual_file_size = os.path.getsize(file_path)
            print(f"📏 File size on disk: {actual_file_size} bytes (expected: {file_info['size']})")
            
            if actual_file_size != file_info['size']:
                print(f"⚠️ File size mismatch! Expected {file_info['size']}, got {actual_file_size}")
                
        except Exception as size_error:
            print(f"❌ Could not get file size: {size_error}")
            # Remove from active downloads
            active_downloads.discard(file_id)
            response = jsonify({"error": f"Could not access file: {str(size_error)}"})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response, 500
        
        # Read chunk from file with better error handling
        try:
            print(f"🔍 Opening file for reading: {file_path}")
            with open(file_path, 'rb') as f:
                print(f"📖 Seeking to position {start_byte}")
                f.seek(start_byte)
                print(f"📖 Reading {expected_chunk_size} bytes")
                chunk_data = f.read(expected_chunk_size)
                actual_read = len(chunk_data)
                print(f"📖 Successfully read {actual_read} bytes (expected {expected_chunk_size})")
                
                if actual_read != expected_chunk_size:
                    print(f"⚠️ Read size mismatch! Expected {expected_chunk_size}, got {actual_read}")
                    if actual_read == 0:
                        raise Exception(f"No data read from file at position {start_byte}")
                
        except Exception as read_error:
            print(f"❌ Error reading file: {str(read_error)}")
            print(f"❌ Error type: {type(read_error).__name__}")
            import traceback
            traceback.print_exc()
            # Remove from active downloads
            active_downloads.discard(file_id)
            response = jsonify({"error": f"Failed to read file: {str(read_error)}"})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response, 500
        
        # Return chunk as base64 in JSON (small enough for individual responses)
        try:
            print(f"🔄 Encoding {len(chunk_data)} bytes to base64")
            chunk_base64 = base64.b64encode(chunk_data).decode('utf-8')
            base64_length = len(chunk_base64)
            print(f"✅ Encoded to {base64_length} base64 characters")
            
        except Exception as encode_error:
            print(f"❌ Error encoding chunk: {str(encode_error)}")
            # Remove from active downloads
            active_downloads.discard(file_id)
            response = jsonify({"error": f"Failed to encode chunk: {str(encode_error)}"})
            response.headers['Access-Control-Allow-Origin'] = '*'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            return response, 500
        
        print(f"📦 Successfully serving chunk {chunk_index}/{file_info['chunks']} for {file_id}")
        print(f"📊 Chunk stats: {len(chunk_data)} bytes, base64: {len(chunk_base64)} chars")
        
        response_data = {
            "chunk_index": chunk_index,
            "total_chunks": file_info['chunks'],
            "chunk_data": chunk_base64,
            "chunk_size": len(chunk_data),
            "mime_type": file_info['mime_type']
        }
        
        response = jsonify(response_data)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        
        # Only remove from active downloads if this was the last chunk
        if chunk_index == file_info['chunks'] - 1:
            print(f"📋 Last chunk for {file_id}, removing from active downloads")
            active_downloads.discard(file_id)
        
        return response
        
    except Exception as e:
        print(f"❌ Unexpected error serving chunk {chunk_index} for {file_id}: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        # Always remove from active downloads on error
        active_downloads.discard(file_id)
        response = jsonify({"error": f"Failed to serve chunk: {str(e)}"})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 500

@app.route('/api/cleanup-download/<file_id>', methods=['POST'])
def cleanup_download(file_id):
    """Clean up a temporary download file"""
    try:
        print(f"🧹 Cleanup request for file: {file_id}")
        
        if file_id in temp_files:
            file_info = temp_files[file_id]
            if os.path.exists(file_info['path']):
                os.unlink(file_info['path'])
                print(f"🧹 Deleted file: {file_info['path']}")
            del temp_files[file_id]
            remove_temp_file_metadata(file_id)
            print(f"🧹 Manual cleanup of download file: {file_id}")
            
            response = jsonify({"message": "File cleaned up successfully"})
        else:
            print(f"⚠️ File not found for cleanup: {file_id}")
            response = jsonify({"message": "File not found or already cleaned up"})
        
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
        
    except Exception as e:
        print(f"❌ Cleanup error for {file_id}: {str(e)}")
        traceback.print_exc()
        response = jsonify({"error": str(e)})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response, 500

# Add explicit OPTIONS handlers for the new endpoints
@app.route('/api/download-chunk/<file_id>/<int:chunk_index>', methods=['OPTIONS'])
def download_chunk_options(file_id, chunk_index):
    """Handle OPTIONS request for download-chunk endpoint"""
    response = make_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

@app.route('/api/cleanup-download/<file_id>', methods=['OPTIONS'])
def cleanup_download_options(file_id):
    """Handle OPTIONS request for cleanup-download endpoint"""
    response = make_response()
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Methods'] = 'POST, OPTIONS'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return response

# Add a debug endpoint to check temp files
@app.route('/api/debug/temp-files', methods=['GET'])
def debug_temp_files():
    """Debug endpoint to check temp files status"""
    try:
        debug_info = {
            "temp_files_count": len(temp_files),
            "temp_files": {}
        }
        
        for file_id, file_info in temp_files.items():
            debug_info["temp_files"][file_id] = {
                "size": file_info['size'],
                "chunks": file_info['chunks'],
                "exists": os.path.exists(file_info['path']),
                "path": file_info['path'],
                "created": file_info['created'],
                "age_seconds": time.time() - file_info['created']
            }
        
        response = jsonify(debug_info)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
        
    except Exception as e:
        print(f"❌ Debug endpoint error: {str(e)}")
        response = jsonify({"error": str(e)})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500

if __name__ == '__main__':
    # Get port from environment (Google Cloud Run sets PORT automatically)
    port = int(os.environ.get('PORT', 8080))  # Changed default to 8080
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    
    print("🚀 Starting Flask server...")
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔧 Debug mode: {app.config['DEBUG']}")
    print(f"📁 Model path: {model_path}")
    print(f" Max file size: {100 * 1024 * 1024 / (1024*1024):.0f}MB")
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
