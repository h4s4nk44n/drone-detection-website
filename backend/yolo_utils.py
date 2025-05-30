# yolo_utils.py
import os
import cv2
from ultralytics import YOLO
import base64
import tempfile
from io import BytesIO
import numpy as np
import yt_dlp
import subprocess

def process_file(file_path, model, session_id=None):
    """
    Process an image or video file with YOLO model
    """
    print(f"🔧 Processing file: {file_path}")
    
    # Get file extension and base info
    file_ext = os.path.splitext(file_path)[1].lower()
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    
    # Create session-specific output filename
    if session_id:
        if file_ext == '.mp4':
            output_name = f'output_{base_name}.webm'
        else:
            output_name = f'output_{base_name}{file_ext}'
    else:
        # Fallback for backward compatibility
        if file_ext == '.mp4':
            output_name = 'output_' + os.path.splitext(os.path.basename(file_path))[0] + '.webm'
        else:
            output_name = 'output_' + os.path.basename(file_path)
    
    output_path = os.path.join('uploads', output_name)
    
    print(f"📤 Output will be saved to: {output_path}")
    print(f"🆔 Session ID: {session_id[:8] if session_id else 'None'}...")
    
    try:
        if file_ext in ['.jpg', '.jpeg', '.png', '.webp']:
            # Process image
            print("🖼️ Processing image...")
            results = model.predict(file_path, verbose=False)
            
            # Use plot method for consistent results
            annotated_img = results[0].plot()
            cv2.imwrite(output_path, annotated_img)
            print(f"✅ Image saved to: {output_path}")
            
        elif file_ext == '.mp4':
            # Always use manual processing for videos with WebM output
            print("🎥 Processing video with manual frame-by-frame method (WebM output)...")
            process_video_manual(file_path, output_path, model)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Verify output file exists and has content
        if not os.path.exists(output_path):
            raise Exception(f"Output file was not created: {output_path}")
        
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            raise Exception(f"Output file is empty: {output_path}")
        
        print(f"✅ Processing complete: {output_path} ({file_size / (1024*1024):.2f} MB)")
        return output_path
        
    except Exception as e:
        print(f"❌ Error in process_file: {str(e)}")
        # Clean up any partial files
        if os.path.exists(output_path):
            os.remove(output_path)
        raise e

def process_video_manual(input_path, output_path, model):
    """
    Process video frame by frame and create WebM output video
    """
    print(f"🎬 Starting manual video processing to WebM format...")
    print(f"📁 Input: {input_path}")
    print(f"📁 Output: {output_path}")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video properties:")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Total frames: {total_frames}")
    
    # Try different codecs for WebM format
    codecs_to_try = [
        ('VP80', 'VP8'),  # VP8 codec for WebM
        ('VP90', 'VP9'),  # VP9 codec for WebM
        ('mp4v', 'MP4V'), # Fallback to MP4V
        ('XVID', 'XVID')  # Another fallback
    ]
    
    out = None
    used_codec = None
    
    for codec_code, codec_name in codecs_to_try:
        print(f"🔧 Trying {codec_name} codec...")
        fourcc = cv2.VideoWriter_fourcc(*codec_code)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if out.isOpened():
            used_codec = codec_name
            print(f"✅ Successfully initialized video writer with {codec_name}")
            break
        else:
            print(f"❌ {codec_name} codec failed")
            if out:
                out.release()
    
    if not out or not out.isOpened():
        cap.release()
        raise Exception(f"Could not create output video writer for: {output_path}")
    
    frame_count = 0
    successful_frames = 0
    
    try:
        print(f"🎬 Starting frame processing with {used_codec} codec...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"📹 End of video reached at frame {frame_count}")
                break
            
            frame_count += 1
            
            # Show progress
            if frame_count == 1 or frame_count % 25 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📹 Frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            try:
                # Run YOLO prediction on this frame
                results = model.predict(frame, verbose=False)
                
                # Get the annotated frame
                annotated_frame = results[0].plot()
                
                # Ensure frame dimensions are correct
                if annotated_frame.shape[:2] != (height, width):
                    annotated_frame = cv2.resize(annotated_frame, (width, height))
                
                # Write the frame to output video
                out.write(annotated_frame)
                successful_frames += 1
                
            except Exception as frame_error:
                print(f"⚠️ Error processing frame {frame_count}: {frame_error}")
                # Write original frame if processing fails
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
        
        print(f"✅ Video processing complete!")
        print(f"   - Total frames processed: {frame_count}")
        print(f"   - Successful AI annotations: {successful_frames}")
        print(f"   - Output codec: {used_codec}")
        
    except Exception as e:
        print(f"❌ Error during video processing: {str(e)}")
        raise e
    finally:
        # Always clean up
        print("🧹 Cleaning up video resources...")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Verify the output file
    if os.path.exists(output_path):
        file_size = os.path.getsize(output_path)
        print(f"📊 Output video created: {file_size / (1024*1024):.2f} MB")
        
        if file_size < 1024:  # Less than 1KB is probably empty
            raise Exception("Output video file is too small, likely empty")
            
        # Test if the video can be opened
        test_cap = cv2.VideoCapture(output_path)
        if test_cap.isOpened():
            test_frames = int(test_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            test_width = int(test_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            test_height = int(test_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            test_cap.release()
            print(f"✅ Output video verified: {test_frames} frames, {test_width}x{test_height}")
        else:
            raise Exception("Output video file cannot be opened")
            
    else:
        raise Exception(f"Output video file was not created: {output_path}")

def process_file_in_memory(file_data, file_ext, filename, model):
    """
    Process an image or video file with YOLO model entirely in memory
    Returns: (original_base64, processed_base64, mime_type)
    """
    print(f"🔧 Processing file in memory: {filename}")
    
    try:
        if file_ext in ['.jpg', '.jpeg', '.png', '.webp']:
            # Process image in memory
            print("🖼️ Processing image in memory...")
            return process_image_in_memory(file_data, file_ext, model)
            
        elif file_ext == '.mp4':
            # Process video in memory
            print("🎥 Processing video in memory...")
            return process_video_in_memory(file_data, model)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
            
    except Exception as e:
        print(f"❌ Error in process_file_in_memory: {str(e)}")
        raise e

def process_image_in_memory(file_data, file_ext, model):
    """
    Process image entirely in memory
    """
    try:
        # Convert bytes to numpy array
        nparr = np.frombuffer(file_data, np.uint8)
        
        # Decode image
        original_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if original_img is None:
            raise Exception("Could not decode image")
        
        print(f"📊 Image shape: {original_img.shape}")
        
        # Run YOLO prediction
        results = model.predict(original_img, verbose=False)
        
        # Get annotated image
        annotated_img = results[0].plot()
        
        # Encode original image to base64
        original_base64 = base64.b64encode(file_data).decode('utf-8')
        
        # Encode processed image to base64
        success, processed_buffer = cv2.imencode(file_ext, annotated_img)
        if not success:
            raise Exception("Could not encode processed image")
        
        processed_base64 = base64.b64encode(processed_buffer).decode('utf-8')
        
        # Determine MIME type
        mime_types = {
            '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
            '.png': 'image/png', '.webp': 'image/webp'
        }
        mime_type = mime_types.get(file_ext, 'image/jpeg')
        
        print("✅ Image processing complete")
        return original_base64, processed_base64, mime_type
        
    except Exception as e:
        print(f"❌ Error processing image: {str(e)}")
        raise e

def process_video_in_memory(file_data, model):
    """
    Process video entirely in memory using temporary files only during processing
    """
    temp_input = None
    temp_output = None
    
    try:
        print("🎬 Starting in-memory video processing...")
        
        # Create temporary files for video processing (OpenCV requirement)
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
            temp_input.write(file_data)
            temp_input_path = temp_input.name
        
        # Create temp output file
        temp_output_path = tempfile.mktemp(suffix='.webm')
        
        try:
            # Process video using temporary files
            process_video_with_temp_files(temp_input_path, temp_output_path, model)
            
            # Read original file back to base64
            original_base64 = base64.b64encode(file_data).decode('utf-8')
            
            # Read processed video to base64
            with open(temp_output_path, 'rb') as f:
                processed_data = f.read()
            processed_base64 = base64.b64encode(processed_data).decode('utf-8')
            
            print("✅ Video processing complete")
            return original_base64, processed_base64, 'video/webm'
            
        finally:
            # Always clean up temporary files
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
            if os.path.exists(temp_output_path):
                os.unlink(temp_output_path)
            print("🧹 Temporary files cleaned up")
            
    except Exception as e:
        print(f"❌ Error processing video: {str(e)}")
        # Clean up on error
        if temp_input and os.path.exists(temp_input.name):
            os.unlink(temp_input.name)
        if temp_output and os.path.exists(temp_output):
            os.unlink(temp_output)
        raise e

def process_video_with_temp_files(input_path, output_path, model):
    """
    Process video frame by frame using temporary files
    """
    print(f"🎬 Processing video with temp files...")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {input_path}")
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"📊 Video properties:")
    print(f"   - Resolution: {width}x{height}")
    print(f"   - FPS: {fps}")
    print(f"   - Total frames: {total_frames}")
    
    # Try different codecs for WebM format
    codecs_to_try = [
        ('VP80', 'VP8'),  # VP8 codec for WebM
        ('VP90', 'VP9'),  # VP9 codec for WebM
        ('mp4v', 'MP4V'), # Fallback to MP4V
        ('XVID', 'XVID')  # Another fallback
    ]
    
    out = None
    used_codec = None
    
    for codec_code, codec_name in codecs_to_try:
        print(f"🔧 Trying {codec_name} codec...")
        fourcc = cv2.VideoWriter_fourcc(*codec_code)
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if out.isOpened():
            used_codec = codec_name
            print(f"✅ Successfully initialized video writer with {codec_name}")
            break
        else:
            print(f"❌ {codec_name} codec failed")
            if out:
                out.release()
    
    if not out or not out.isOpened():
        cap.release()
        raise Exception(f"Could not create output video writer")
    
    frame_count = 0
    successful_frames = 0
    
    try:
        print(f"🎬 Starting frame processing with {used_codec} codec...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"📹 End of video reached at frame {frame_count}")
                break
            
            frame_count += 1
            
            # Show progress
            if frame_count == 1 or frame_count % 25 == 0:
                progress = (frame_count / total_frames) * 100
                print(f"📹 Frame {frame_count}/{total_frames} ({progress:.1f}%)")
            
            try:
                # Run YOLO prediction on this frame
                results = model.predict(frame, verbose=False)
                
                # Get the annotated frame
                annotated_frame = results[0].plot()
                
                # Ensure frame dimensions are correct
                if annotated_frame.shape[:2] != (height, width):
                    annotated_frame = cv2.resize(annotated_frame, (width, height))
                
                # Write the frame to output video
                out.write(annotated_frame)
                successful_frames += 1
                
            except Exception as frame_error:
                print(f"⚠️ Error processing frame {frame_count}: {frame_error}")
                # Write original frame if processing fails
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                out.write(frame)
        
        print(f"✅ Video processing complete!")
        print(f"   - Total frames processed: {frame_count}")
        print(f"   - Successful AI annotations: {successful_frames}")
        print(f"   - Output codec: {used_codec}")
        
    except Exception as e:
        print(f"❌ Error during video processing: {str(e)}")
        raise e
    finally:
        # Always clean up
        print("🧹 Cleaning up video resources...")
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    # Verify the output file was created
    if not os.path.exists(output_path):
        raise Exception("Output video file was not created")
    
    file_size = os.path.getsize(output_path)
    if file_size < 1024:  # Less than 1KB is probably empty
        raise Exception("Output video file is too small, likely empty")
    
    print(f"📊 Output video created: {file_size / (1024*1024):.2f} MB")

def process_youtube_video(youtube_url, model):
    """
    Download and process YouTube video entirely in memory
    Returns: (original_base64, processed_base64)
    """
    temp_download = None
    temp_converted = None
    temp_output = None
    
    try:
        print(f"🔗 Starting YouTube video processing: {youtube_url}")
        
        # Create temporary file for download
        temp_download = tempfile.mktemp(suffix='.%(ext)s')
        temp_converted = tempfile.mktemp(suffix='.mp4')
        temp_output = tempfile.mktemp(suffix='.webm')
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'best[height<=720][ext=mp4]/best[height<=720]/best[ext=mp4]/best',  # Prefer MP4, max 720p
            'outtmpl': temp_download,
            'quiet': False,
            'no_warnings': False,
            'extractaudio': False,
            'audioformat': 'mp3',
            'embed_subs': False,
            'writesubtitles': False,
            'writeautomaticsub': False,
        }
        
        print("📥 Downloading YouTube video...")
        
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Get video info first
            info = ydl.extract_info(youtube_url, download=False)
            title = info.get('title', 'Unknown')
            duration = info.get('duration', 0)
            
            print(f"📹 Video title: {title}")
            print(f"⏱️ Duration: {duration}s")
            
            # Check duration limit (10 minutes = 600 seconds)
            if duration and duration > 600:
                raise Exception("Video too long! Maximum duration is 10 minutes.")
            
            # Download the video
            ydl.download([youtube_url])
        
        # Find the downloaded file (yt-dlp might change the extension)
        downloaded_files = []
        for ext in ['.mp4', '.webm', '.mkv', '.avi']:
            potential_file = temp_download.replace('.%(ext)s', ext)
            if os.path.exists(potential_file):
                downloaded_files.append(potential_file)
        
        if not downloaded_files:
            raise Exception("Downloaded video file not found")
        
        actual_download_path = downloaded_files[0]
        print(f"✅ Video downloaded: {os.path.basename(actual_download_path)}")
        
        # Convert to MP4 if needed (for consistent processing)
        if not actual_download_path.endswith('.mp4'):
            print("🔄 Converting video to MP4 format...")
            cmd = [
                'ffmpeg', '-i', actual_download_path,
                '-c:v', 'libx264', '-c:a', 'aac',
                '-y', temp_converted
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"⚠️ FFmpeg conversion failed, using original file")
                temp_converted = actual_download_path
            else:
                print("✅ Video converted to MP4")
        else:
            temp_converted = actual_download_path
        
        # Check file size
        file_size = os.path.getsize(temp_converted)
        print(f"📊 Video file size: {file_size / (1024*1024):.2f} MB")
        
        if file_size > 100 * 1024 * 1024:  # 100MB limit
            raise Exception("Downloaded video too large! Maximum size is 100MB.")
        
        # Read original video to base64
        with open(temp_converted, 'rb') as f:
            original_data = f.read()
        original_base64 = base64.b64encode(original_data).decode('utf-8')
        
        # Process video with YOLO
        print("🤖 Processing video with YOLO...")
        process_video_with_temp_files(temp_converted, temp_output, model)
        
        # Read processed video to base64
        with open(temp_output, 'rb') as f:
            processed_data = f.read()
        processed_base64 = base64.b64encode(processed_data).decode('utf-8')
        
        print("✅ YouTube video processing complete")
        return original_base64, processed_base64
        
    except Exception as e:
        print(f"❌ Error processing YouTube video: {str(e)}")
        raise e
        
    finally:
        # Clean up all temporary files
        temp_files = [temp_download, temp_converted, temp_output]
        for temp_file in temp_files:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.unlink(temp_file)
                    print(f"🧹 Cleaned up: {os.path.basename(temp_file)}")
                except:
                    pass
        
        # Also clean up any files that might have different extensions
        if temp_download:
            base_path = temp_download.replace('.%(ext)s', '')
            for ext in ['.mp4', '.webm', '.mkv', '.avi', '.flv']:
                potential_file = base_path + ext
                if os.path.exists(potential_file):
                    try:
                        os.unlink(potential_file)
                        print(f"🧹 Cleaned up: {os.path.basename(potential_file)}")
                    except:
                        pass
