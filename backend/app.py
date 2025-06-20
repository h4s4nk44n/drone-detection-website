from flask import Flask, request, jsonify, make_response
import os
import base64
from ultralytics import YOLO
from yolo_utils import process_file_in_memory, process_youtube_video, log_memory_usage, cleanup_memory
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
import zipfile
import yaml

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
    """Load saved file metadata from disk on startup"""
    temp_dir = tempfile.gettempdir()
    loaded = 0
    
    try:
        for item in os.listdir(temp_dir):
            if item.endswith('_metadata.json'):
                file_id = item.replace('_metadata.json', '')
                try:
                    with open(os.path.join(temp_dir, item), 'r') as f:
                        file_info = json.load(f)
                        
                    # Convert downloaded_chunks back to set
                    if 'downloaded_chunks' in file_info:
                        file_info['downloaded_chunks'] = set(file_info['downloaded_chunks'])
                    else:
                        file_info['downloaded_chunks'] = set()
                        
                    if os.path.exists(file_info['path']):
                        temp_files[file_id] = file_info
                        loaded += 1
                        print(f"📂 Loaded {file_id} from disk")
                except Exception as e:
                    print(f"⚠️ Error loading {item}: {e}")
                    
        print(f"✅ Loaded {loaded} files from disk")
    except Exception as e:
        print(f"⚠️ Error listing temp directory: {e}")
        
    return loaded

def save_temp_file_metadata(file_id, file_info):
    """Save file metadata to disk for persistence across workers"""
    try:
        # Initialize downloaded_chunks if not present
        if 'downloaded_chunks' not in file_info:
            file_info['downloaded_chunks'] = set()
            
        # Convert set to list for JSON serialization
        file_info_json = file_info.copy()
        file_info_json['downloaded_chunks'] = list(file_info['downloaded_chunks'])
        
        metadata_path = os.path.join(tempfile.gettempdir(), f"{file_id}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(file_info_json, f)
        print(f"💾 Saved metadata for {file_id}")
    except Exception as e:
        print(f"⚠️ Error saving metadata for {file_id}: {e}")

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
    """Clean up files older than 4 hours (increased from 2 hours for chunked downloads)"""
    current_time = time.time()
    expired_files = []
    
    temp_dir_path = tempfile.gettempdir() # Get temp dir path once

    # First, get all file_ids from temp_files and potentially from orphaned metadata
    all_known_file_ids = set(temp_files.keys())
    try:
        for item_name in os.listdir(temp_dir_path):
            if item_name.endswith('_metadata.json'):
                all_known_file_ids.add(item_name.replace('_metadata.json', ''))
    except OSError as e:
        print(f"⚠️ Error listing temp directory for cleanup: {e}")

    for file_id in list(all_known_file_ids): # Iterate over a copy if modifying temp_files
        # Check for filesystem lock first, most robust cross-worker
        lock_file_path = os.path.join(temp_dir_path, f"{file_id}.active_download_lock")
        if os.path.exists(lock_file_path):
            try:
                lock_age = current_time - os.path.getmtime(lock_file_path)
                if lock_age > 3600: # 60 minutes (increased from 30)
                    print(f"🔒 Removing stale lock file for {file_id} (age: {lock_age:.0f}s)")
                    os.unlink(lock_file_path)
                else:
                    print(f"🔒 Skipping cleanup of {file_id} due to active lock file (age: {lock_age:.0f}s).")
                    continue
            except OSError: # File might have been removed between exists() and getmtime()
                pass # Proceed with other checks

        # If not locked, proceed with existing logic (in-memory active_downloads and last_accessed)
        if file_id in active_downloads:
            print(f"⏳ Skipping cleanup of active download (in-memory): {file_id}")
            continue

        file_info = temp_files.get(file_id)
        if not file_info:
            # Try to load from disk metadata
            try:
                metadata_path = os.path.join(temp_dir_path, f"{file_id}_metadata.json")
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r') as f:
                        file_info = json.load(f)
                        # Convert downloaded_chunks back to set if present
                        if 'downloaded_chunks' in file_info:
                            file_info['downloaded_chunks'] = set(file_info['downloaded_chunks'])
                        else:
                            file_info['downloaded_chunks'] = set()
            except Exception as e:
                print(f"⚠️ Error loading metadata for {file_id}: {e}")
                continue

        if file_info:
            file_age = current_time - file_info.get('created', 0)
            last_accessed_age = current_time - file_info.get('last_accessed', 0)
            
            # Check if all chunks have been downloaded
            total_chunks = file_info.get('chunks', 0)
            downloaded_chunks = len(file_info.get('downloaded_chunks', set()))
            all_chunks_downloaded = downloaded_chunks >= total_chunks
            
            print(f"📊 File {file_id} status:")
            print(f"   - Age: {file_age:.0f}s")
            print(f"   - Last accessed: {last_accessed_age:.0f}s ago")
            print(f"   - Chunks downloaded: {downloaded_chunks}/{total_chunks}")
            
            # More conservative cleanup timing AND require all chunks downloaded
            if file_age > 14400: # 4 hours (increased from 2)
                if last_accessed_age > 7200 and all_chunks_downloaded: # 2 hours since last access AND all chunks downloaded
                    print(f"🕒 File {file_id} expired and all chunks downloaded")
                    expired_files.append(file_id)
                elif all_chunks_downloaded:
                    print(f"⏳ File {file_id} old but recently accessed (all chunks downloaded)")
                else:
                    print(f"⏳ File {file_id} old but chunks still downloading ({downloaded_chunks}/{total_chunks})")
            else:
                print(f"✅ File {file_id} still valid: age={file_age:.0f}s")

    # Clean up expired files
    for file_id in expired_files:
        try:
            # Double check lock file one last time before deleting
            lock_file_path = os.path.join(temp_dir_path, f"{file_id}.active_download_lock")
            if os.path.exists(lock_file_path):
                lock_age = current_time - os.path.getmtime(lock_file_path)
                if lock_age <= 3600: # 60 minutes
                    print(f"🔒 Final check: Skipping delete of {file_id} due to active lock file (age: {lock_age:.0f}s).")
                    continue
                else:
                    print(f"🔒 Final check: Removing stale lock for {file_id} before deleting data.")
                    os.unlink(lock_file_path)

            file_info = temp_files.get(file_id)
            if file_info and 'path' in file_info:
                # Final check of downloaded chunks
                total_chunks = file_info.get('chunks', 0)
                downloaded_chunks = len(file_info.get('downloaded_chunks', set()))
                if downloaded_chunks < total_chunks:
                    print(f"⚠️ Skipping deletion of {file_id} - not all chunks downloaded ({downloaded_chunks}/{total_chunks})")
                    continue
                    
                file_path = file_info['path']
                if os.path.exists(file_path):
                    os.unlink(file_path)
                    print(f"🧹 Cleaned up expired file: {file_path}")
                del temp_files[file_id]
            
            # Always try to clean up metadata file
            metadata_path = os.path.join(temp_dir_path, f"{file_id}_metadata.json")
            if os.path.exists(metadata_path):
                os.unlink(metadata_path)
                print(f"🧹 Cleaned up metadata file: {metadata_path}")
                
        except Exception as e:
            print(f"⚠️ Error cleaning up {file_id}: {e}")
            continue # Continue with other files even if one fails

    return len(expired_files)

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
        log_memory_usage("upload start")
        
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
            
            # Clean up memory before returning
            cleanup_memory()
            log_memory_usage("upload success - direct response")
            
            return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error in upload_file: {str(e)}")
        traceback.print_exc()
        
        # Clean up memory on error
        cleanup_memory()
        log_memory_usage("upload error")
        
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
    lock_file_path = os.path.join(tempfile.gettempdir(), f"{file_id}.active_download_lock")
    created_lock_this_request = False
    max_retries = 5  # Increased from 3
    retry_delay = 2  # Increased from 1 second
    
    try:
        print(f"📥 Download request: file_id={file_id}, chunk={chunk_index}")
        
        # Create/touch lock file to show activity
        if not os.path.exists(lock_file_path):
            open(lock_file_path, 'a').close()
            created_lock_this_request = True
            print(f"🔒 Created lock file: {lock_file_path}")
        else:
            # Touch the lock file to update its modification time
            os.utime(lock_file_path, None) 
            print(f"🔒 Touched lock file: {lock_file_path}")

        # Mark file as actively being downloaded (in-memory for this worker)
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
            # Wait up to 30 seconds for file to be ready (increased from 10)
            wait_time = 0
            while not file_info.get('ready', False) and wait_time < 30:
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
        
        # Implement retries for file access with exponential backoff
        last_error = None
        chunk_data = None
        
        for retry in range(max_retries):
            try:
                # Check file size on disk before reading
                actual_file_size = os.path.getsize(file_path)
                print(f"📏 File size on disk: {actual_file_size} bytes (expected: {file_info['size']})")
                
                if actual_file_size != file_info['size']:
                    print(f"⚠️ File size mismatch! Expected {file_info['size']}, got {actual_file_size}")
                    if retry < max_retries - 1:
                        wait_time = retry_delay * (2 ** retry)  # Exponential backoff
                        print(f"🔄 Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                
                # Read chunk from file with explicit error handling
                try:
                    with open(file_path, 'rb') as f:
                        f.seek(start_byte)
                        chunk_data = f.read(expected_chunk_size)
                        actual_read = len(chunk_data)
                        
                        if actual_read == 0:
                            raise Exception(f"No data read from file at position {start_byte}")
                        
                        if actual_read != expected_chunk_size:
                            print(f"⚠️ Read size mismatch! Expected {expected_chunk_size}, got {actual_read}")
                            if retry < max_retries - 1:
                                wait_time = retry_delay * (2 ** retry)  # Exponential backoff
                                print(f"🔄 Retrying in {wait_time} seconds...")
                                time.sleep(wait_time)
                                continue
                        
                        # If we got here, the read was successful
                        print(f"✅ Successfully read {actual_read} bytes on attempt {retry + 1}")
                        break
                except IOError as io_error:
                    print(f"⚠️ IO Error reading file: {io_error}")
                    if retry < max_retries - 1:
                        wait_time = retry_delay * (2 ** retry)  # Exponential backoff
                        print(f"🔄 Retrying in {wait_time} seconds...")
                        time.sleep(wait_time)
                        continue
                    raise
                    
            except Exception as e:
                last_error = e
                print(f"❌ Error reading file (attempt {retry + 1}): {str(e)}")
                if retry < max_retries - 1:
                    wait_time = retry_delay * (2 ** retry)  # Exponential backoff
                    print(f"🔄 Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print("❌ All retries failed")
                    raise
        
        if chunk_data is None:
            raise Exception(f"Failed to read chunk after {max_retries} attempts: {str(last_error)}")
        
        # Convert to base64
        try:
            chunk_base64 = base64.b64encode(chunk_data).decode('utf-8')
            print(f"✅ Encoded {len(chunk_data)} bytes to {len(chunk_base64)} base64 chars")
            
        except Exception as encode_error:
            print(f"❌ Error encoding chunk: {str(encode_error)}")
            raise
        
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
            if os.path.exists(lock_file_path):
                try:
                    os.unlink(lock_file_path)
                    print(f"🔒 Removed lock file on completion: {lock_file_path}")
                except OSError as e:
                    print(f"⚠️ Error removing lock file {lock_file_path} on completion: {e}")
        
        # After successful chunk read and before returning response
        if file_id in temp_files:
            # Track this chunk as downloaded
            if 'downloaded_chunks' not in temp_files[file_id]:
                temp_files[file_id]['downloaded_chunks'] = set()
            temp_files[file_id]['downloaded_chunks'].add(chunk_index)
            
            # Save updated metadata
            save_temp_file_metadata(file_id, temp_files[file_id])
            
            print(f"✅ Marked chunk {chunk_index} as downloaded for {file_id}")
            print(f"📊 Downloaded chunks: {len(temp_files[file_id]['downloaded_chunks'])}/{file_info['chunks']}")
        
        return response
        
    except Exception as e:
        print(f"❌ Unexpected error serving chunk {chunk_index} for {file_id}: {str(e)}")
        print(f"❌ Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
        # Always remove from active downloads on error for this worker
        active_downloads.discard(file_id)
        
        # Clean up lock file if appropriate
        if created_lock_this_request or isinstance(e, FileNotFoundError) or "File not found" in str(e):
            if os.path.exists(lock_file_path):
                try:
                    os.unlink(lock_file_path)
                    print(f"🔒 Removed lock file due to error: {lock_file_path}")
                except OSError as unlink_e:
                    print(f"⚠️ Error removing lock file {lock_file_path}: {unlink_e}")

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

@app.route('/api/debug/memory', methods=['GET'])
def debug_memory():
    """Debug endpoint to check memory usage"""
    try:
        memory_mb = log_memory_usage("debug endpoint")
        process = psutil.Process()
        memory_info = process.memory_info()
        system_memory = psutil.virtual_memory()
        
        debug_info = {
            "process_memory_mb": memory_info.rss / (1024 * 1024),
            "system_memory_total_mb": system_memory.total / (1024 * 1024),
            "system_memory_available_mb": system_memory.available / (1024 * 1024),
            "system_memory_percent": system_memory.percent,
            "active_temp_files": len(temp_files),
            "active_downloads": len(active_downloads),
            "memory_warning": memory_info.rss / (1024 * 1024) > 3000
        }
        
        response = jsonify(debug_info)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
        
    except Exception as e:
        print(f"❌ Memory debug endpoint error: {str(e)}")
        response = jsonify({"error": str(e)})
        response.headers['Access-Control-Allow-Origin'] = '*'
        return response, 500

@app.route('/api/extract-training-data', methods=['POST'])
def extract_training_data():
    """Extract YOLO format training data from uploaded file"""
    try:
        print("🎯 === TRAINING DATA EXTRACTION ENDPOINT REACHED ===")
        
        # Cleanup old files first
        cleanup_old_files()
        
        if 'file' not in request.files:
            print("❌ No file in request")
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            print("❌ Empty filename")
            return jsonify({"error": "No file selected"}), 400
        
        print(f"📁 File received: {file.filename}")
        
        # Read file data
        file_data = file.read()
        file_size_mb = len(file_data) / (1024 * 1024)
        print(f"📊 File size: {file_size_mb:.2f}MB")
        
        # File size check
        if file_size_mb > 100:  # Increased limit for training data extraction
            print(f"❌ File too large: {file_size_mb:.2f}MB")
            return jsonify({"error": f"File too large. Maximum size is 100MB for training data extraction."}), 400
        
        print(f"📁 File uploaded for training extraction: {file.filename} ({file_size_mb:.2f}MB)")
        
        # Get file extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        print(f"📋 File extension: {file_ext}")
        
        # File type validation
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.webm'}
        if file_ext not in allowed_extensions:
            print(f"❌ Unsupported file type: {file_ext}")
            return jsonify({"error": f"Unsupported file type: {file_ext}"}), 400
        
        # Get confidence threshold from request (optional)
        confidence_threshold = float(request.form.get('confidence', 0.3))
        if confidence_threshold < 0.1 or confidence_threshold > 0.9:
            confidence_threshold = 0.3
        
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        print(f"🎯 Starting training data extraction...")
        
        # Process file and extract training data using the existing processing function
        print("🤖 Calling process_file_in_memory with training data extraction enabled...")
        
        try:
            # Use the modified process_file_in_memory with training data extraction enabled
            result = process_file_in_memory(
                file_data, file_ext, file.filename, model, 
                extract_training_data=True, 
                confidence_threshold=confidence_threshold
            )
            
            print(f"✅ process_file_in_memory completed successfully")
            print(f"📊 Result type: {type(result)}")
            print(f"📊 Result length: {len(result) if hasattr(result, '__len__') else 'N/A'}")
            
            # Unpack the result
            if len(result) == 6: # This is the expected case when extract_training_data=True
                original_base64, processed_base64, mime_type, training_images_data, training_labels_data, extracted_count = result
                print(f"🎯 Training data extracted: {extracted_count} samples")

                if extracted_count > 0:
                    # If training data was extracted, it's raw images and labels. We need to create the ZIP.
                    print(f"📦 Creating ZIP for {extracted_count} samples for {file.filename}")
                    zip_data = create_combined_training_dataset_zip(training_images_data, training_labels_data, file.filename)
                    print(f"📦 ZIP data size: {len(zip_data) if zip_data else 0} bytes")
                else:
                    zip_data = None # No data extracted, so no ZIP
                    print(f"⚠️ No actual training samples extracted, though 6 values returned (empty lists).")

            elif len(result) == 3: # This case should not be hit if extract_training_data=True, but handle for robustness
                original_base64, processed_base64, mime_type = result
                zip_data = None
                extracted_count = 0
                print(f"⚠️ No training data in result - only {len(result)} elements returned (extract_training_data was likely False or error occurred before data formation).")
            else: # Unexpected number of results
                raise ValueError(f"Unexpected number of values ({len(result)}) returned from process_file_in_memory.")

            print(f"📋 Final extracted count: {extracted_count}")
            
            if extracted_count == 0 or zip_data is None: # Check if zip_data is also None
                print(f"⚠️ No training data extracted or ZIP not created - preparing detailed error response")
                # Enhanced error message with debugging info
                debug_info = {
                    "error": "No drone detections found meeting the confidence criteria.",
                    "debug_info": {
                        "confidence_threshold_used": confidence_threshold,
                        "file_type": file_ext,
                        "file_size_mb": round(file_size_mb, 2),
                        "processing_notes": [
                            "Model uses conf=0.1 to find all detections",
                            f"Then filters by your threshold: {confidence_threshold}",
                            "Check server logs for detailed detection info per frame"
                        ]
                    },
                    "suggestions": [
                        "Try a lower confidence threshold (e.g., 0.1 or 0.2)",
                        "Ensure the video contains clearly visible drones",
                        "Check that the drone is not too small in the frame",
                        "Verify the video quality is good enough for detection"
                    ]
                }
                print(f"❌ Returning 400 error with debug info")
                return jsonify(debug_info), 400
            
            print(f"✅ Training data extraction will proceed with {extracted_count} samples")
            
            # Convert ZIP to base64 for download
            zip_base64 = base64.b64encode(zip_data).decode('utf-8')
            
            return jsonify({
                "message": f"Training data extracted successfully: {extracted_count} samples",
                "extracted_count": extracted_count,
                "confidence_threshold": confidence_threshold,
                "file_size_mb": round(file_size_mb, 2),
                "dataset_zip": f"data:application/zip;base64,{zip_base64}",
                "dataset_size_mb": round(len(zip_data) / (1024 * 1024), 2)
            })
            
        except Exception as processing_error:
            print(f"❌ process_file_in_memory threw an exception: {str(processing_error)}")
            print(f"❌ Exception type: {type(processing_error).__name__}")
            import traceback
            print(f"❌ Full traceback:")
            traceback.print_exc()
            
            # Enhanced error messages
            error_msg = str(processing_error)
            if "No drone detections found" in error_msg:
                return jsonify({
                    "error": "No drone detections found in the uploaded file.",
                    "suggestion": "Make sure your file contains visible drones, or try lowering the confidence threshold.",
                    "confidence_used": confidence_threshold
                }), 400
            elif "Could not open video" in error_msg:
                return jsonify({
                    "error": "Unable to process the video file.",
                    "suggestion": "Try uploading in a different video format (MP4 recommended)."
                }), 400
            elif "Could not load image" in error_msg:
                return jsonify({
                    "error": "Unable to process the image file.",
                    "suggestion": "Try uploading in a different image format (JPG/PNG recommended)."
                }), 400
            else:
                return jsonify({
                    "error": f"Training data extraction failed: {error_msg}"
                }), 500
        
    except Exception as e:
        print(f"❌ Error in extract_training_data endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Server error: {str(e)}"}), 500

@app.route('/api/extract-training-data-bulk', methods=['POST'])
def extract_training_data_bulk():
    """Extract YOLO format training data from multiple uploaded files (images and videos) and combine into one dataset"""
    try:
        print("🎯 === BULK TRAINING DATA EXTRACTION ENDPOINT REACHED (IMAGES & VIDEOS) ===")
        
        cleanup_old_files()
        
        if 'files' not in request.files:
            print("❌ No files in request")
            return jsonify({"error": "No files uploaded"}), 400
        
        files = request.files.getlist('files')
        if not files or len(files) == 0:
            print("❌ Empty files list")
            return jsonify({"error": "No files selected"}), 400
        
        print(f"📁 Files received: {len(files)} files")
        for file_obj in files: # Renamed to avoid conflict with os.path.file
            if file_obj.filename:
                print(f"   - {file_obj.filename}")
        
        confidence_threshold = float(request.form.get('confidence', 0.3))
        if confidence_threshold < 0.1 or confidence_threshold > 0.9:
            confidence_threshold = 0.3
        
        print(f"🎯 Confidence threshold: {confidence_threshold}")
        
        all_training_images = []
        all_training_labels = []
        total_extracted_count = 0
        files_with_detections = []
        files_without_detections = []
        
        for i, file_obj in enumerate(files):
            if not file_obj.filename:
                continue
                
            try:
                print(f"🔄 Processing file {i+1}/{len(files)}: {file_obj.filename}")
                
                file_data = file_obj.read()
                file_size_mb = len(file_data) / (1024 * 1024)
                
                # Adjusted file size limit for individual files in bulk, especially for videos
                # Videos can be larger, but will be processed frame by frame.
                # Images should still be reasonably sized.
                file_ext = os.path.splitext(file_obj.filename)[1].lower()
                is_video = file_ext in ['.mp4', '.avi', '.mov', '.webm']
                max_size_mb = 75 if is_video else 25 # 75MB for videos, 25MB for images in bulk training

                if file_size_mb > max_size_mb:
                    print(f"⚠️ Skipping {file_obj.filename}: too large ({file_size_mb:.2f}MB). Max is {max_size_mb}MB for its type.")
                    files_without_detections.append(f"{file_obj.filename} (too large)")
                    continue
                
                # File type validation - allow images and videos
                allowed_extensions = {'.jpg', '.jpeg', '.png', '.mp4', '.avi', '.mov', '.webm'}
                if file_ext not in allowed_extensions:
                    print(f"⚠️ Skipping {file_obj.filename}: unsupported type {file_ext}")
                    files_without_detections.append(f"{file_obj.filename} (unsupported type)")
                    continue
                
                print(f"📊 Processing {file_obj.filename} ({file_size_mb:.2f}MB, type: {file_ext})")
                
                # Call the modified process_file_in_memory
                # Expecting: original_base64, processed_base64, mime_type, p_training_images, p_training_labels, p_extracted_count
                result = process_file_in_memory(
                    file_data, file_ext, file_obj.filename, model, 
                    extract_training_data=True, 
                    confidence_threshold=confidence_threshold
                )
                
                # Unpack results carefully based on length (to handle cases where not all 6 are returned, e.g. error or no extraction attempted)
                if len(result) == 6:
                    _, _, _, p_training_images, p_training_labels, p_extracted_count = result
                elif len(result) == 3: # Fallback if only base results returned (e.g. non-extraction mode or error before extraction data formed)
                    p_training_images, p_training_labels, p_extracted_count = [], [], 0
                    print(f"⚠️ {file_obj.filename}: Received 3 values from process_file_in_memory, expecting 6 for training. Assuming no detections.")
                else:
                    print(f"❌ {file_obj.filename}: Unexpected number of values ({len(result)}) from process_file_in_memory. Skipping.")
                    files_without_detections.append(f"{file_obj.filename} (processing error)")
                    continue # Skip this file

                if p_extracted_count > 0 and p_training_images and p_training_labels:
                    print(f"✅ {file_obj.filename}: {p_extracted_count} samples extracted.")
                    all_training_images.extend(p_training_images)
                    all_training_labels.extend(p_training_labels)
                    total_extracted_count += p_extracted_count
                    files_with_detections.append(file_obj.filename)
                else:
                    print(f"⚠️ {file_obj.filename}: No detections found or no training data returned.")
                    files_without_detections.append(file_obj.filename)
                    
            except Exception as file_error:
                print(f"❌ Error processing {file_obj.filename}: {str(file_error)}")
                traceback.print_exc() # Print full traceback for file processing errors
                files_without_detections.append(f"{file_obj.filename} (error: {str(file_error)[:50]}...)")
                continue
            finally:
                # Explicitly clear large data from loop to help memory, if possible
                if 'file_data' in locals(): del file_data
                if 'result' in locals(): del result
                if 'p_training_images' in locals(): del p_training_images
                if 'p_training_labels' in locals(): del p_training_labels
                gc.collect()
        
        print(f"📊 Bulk extraction summary:")
        print(f"   - Total samples accumulated: {total_extracted_count}")
        print(f"   - Files with detections: {len(files_with_detections)}")
        print(f"   - Files without detections/errors: {len(files_without_detections)}")
        
        if total_extracted_count == 0:
            return jsonify({
                "error": "No drone detections found in any of the uploaded files meeting the criteria.",
                "debug_info": {
                    "confidence_threshold_used": confidence_threshold,
                    "files_processed_count": len(files),
                    "files_with_detections_list": files_with_detections,
                    "files_without_detections_list": files_without_detections
                },
                "suggestions": [
                    f"Try a lower confidence threshold (current: {confidence_threshold})",
                    "Ensure the files contain clearly visible drones."
                ]
            }), 400
        
        # Create combined training dataset ZIP
        print(f"📦 Creating combined ZIP for {total_extracted_count} samples from {len(all_training_images)} images/frames.")
        combined_zip_data = create_combined_training_dataset_zip(
            all_training_images, 
            all_training_labels, 
            f"bulk_dataset_{len(files_with_detections)}files"
        )
        
        # Convert ZIP to base64 for download
        zip_base64 = base64.b64encode(combined_zip_data).decode('utf-8')
        dataset_size_mb = round(len(combined_zip_data) / (1024 * 1024), 2)

        # Clean up large lists from memory after use
        del all_training_images
        del all_training_labels
        del combined_zip_data
        gc.collect()
        
        return jsonify({
            "message": f"Bulk training data extracted: {total_extracted_count} samples from {len(files_with_detections)} files",
            "extracted_count": total_extracted_count,
            "files_with_detections": files_with_detections,
            "files_without_detections": files_without_detections,
            "confidence_threshold": confidence_threshold,
            "dataset_zip": f"data:application/zip;base64,{zip_base64}",
            "dataset_size_mb": dataset_size_mb
        })
        
    except Exception as e:
        print(f"❌ Error in extract_training_data_bulk endpoint: {str(e)}")
        traceback.print_exc()
        return jsonify({"error": f"Bulk training data extraction failed: {str(e)}"}), 500

def create_combined_training_dataset_zip(training_images, training_labels, dataset_name):
    """
    Create a ZIP file containing combined training images and YOLO format labels from multiple sources
    """
    print(f"📦 Creating combined training dataset ZIP with {len(training_images)} samples...")
    
    # Create in-memory ZIP file
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create images and labels directories
        for i, (image, label) in enumerate(zip(training_images, training_labels)):
            # Create unique filenames for combined dataset
            image_filename = f"images/{dataset_name}_{i:06d}.jpg"
            label_filename = f"labels/{dataset_name}_{i:06d}.txt"
            
            # Save image
            _, buffer = cv2.imencode('.jpg', image)
            zip_file.writestr(image_filename, buffer.tobytes())
            
            # Save label
            zip_file.writestr(label_filename, label)
        
        # Create dataset.yaml file for YOLO training
        dataset_config = {
            'path': './dataset',
            'train': 'images',
            'val': 'images',  # Same as train for this extracted dataset
            'nc': 1,  # Number of classes (assuming drone detection)
            'names': ['drone']  # Class names
        }
        
        yaml_content = yaml.dump(dataset_config, default_flow_style=False)
        zip_file.writestr('dataset.yaml', yaml_content)
        
        # Create README file
        readme_content = f"""# Combined Drone Detection Training Dataset

This dataset was automatically extracted from multiple image files.

## Contents:
- images/: Combined training images with drone detections ({len(training_images)} files)
- labels/: Combined YOLO format annotation files ({len(training_labels)} files)
- dataset.yaml: Dataset configuration for YOLO training

## Dataset Statistics:
- Total samples: {len(training_images)}
- Image format: JPG
- Label format: YOLO (class_id center_x center_y width height - normalized coordinates)
- Filename pattern: {dataset_name}_XXXXXX.jpg/.txt

## YOLO Format Details:
Each .txt file contains one line per detection:
- class_id: 0 (drone)
- center_x: Normalized x-coordinate of bounding box center (0.0-1.0)
- center_y: Normalized y-coordinate of bounding box center (0.0-1.0)
- width: Normalized width of bounding box (0.0-1.0)
- height: Normalized height of bounding box (0.0-1.0)

## Usage:
1. Extract this ZIP file
2. Use with YOLO training frameworks like Ultralytics YOLOv11
3. Update dataset.yaml paths as needed for your environment

Example training command:
```bash
yolo train data=dataset.yaml model=yolo11n.pt epochs=100 imgsz=640
```

## Classes:
- 0: drone

Generated by Drone Detection System - Bulk Extraction
Extracted on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        zip_file.writestr('README.md', readme_content)
    
    zip_data = zip_buffer.getvalue()
    zip_buffer.close()
    
    print(f"✅ Combined training dataset ZIP created: {len(zip_data) / (1024*1024):.2f}MB")
    return zip_data

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
