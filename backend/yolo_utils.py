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
import gc

YOLO_PROCESSING_SIZE = 832  # High accuracy processing for all media

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
    Process file in memory with proper MIME types for web compatibility
    """
    print(f"🔄 Processing file in memory: {filename}")
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_input:
        temp_input.write(file_data)
        temp_input_path = temp_input.name
    
    try:
        if file_ext.lower() in ['.mp4', '.avi', '.mov', '.webm']:
            # Define the output path
            temp_output_path = os.path.splitext(temp_input_path)[0] + '_processed.mp4'
            
            # Get video info first
            cap = cv2.VideoCapture(temp_input_path)
            if not cap.isOpened():
                raise Exception(f"Could not open video: {temp_input_path}")
            
            # Get video properties
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0
            
            print(f"📹 Video info:")
            print(f"   - Resolution: {width}x{height}")
            print(f"   - FPS: {fps}")
            print(f"   - Duration: {duration:.1f}s")
            print(f"   - Total frames: {total_frames}")
            
            # Calculate target resolution (max 720p)
            if width > height:
                output_width = min(width, 1280)
                output_height = int(height * (output_width / width))
            else:
                output_height = min(height, 720)
                output_width = int(width * (output_height / height))
            
            # Ensure even dimensions
            output_width = output_width - (output_width % 2)
            output_height = output_height - (output_height % 2)
            
            # Target lower FPS for processing efficiency
            target_fps = min(30, fps)
            
            print(f"🎯 Processing plan:")
            print(f"   - Output: {output_width}x{output_height} @ {target_fps}fps")
            print(f"   - Scaling: {width}x{height} -> {output_width}x{output_height}")
            
            # Initialize video writer with H.264 codec
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_output_path, fourcc, target_fps, (output_width, output_height))
            
            if not out.isOpened():
                raise Exception("Failed to create video writer")
            
            frame_count = 0
            processed_count = 0
            chunk_size = 30  # Process in chunks of 30 frames
            
            try:
                while True:
                    frames_chunk = []
                    # Read chunk_size frames
                    for _ in range(chunk_size):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_count += 1
                        frames_chunk.append(frame)
                    
                    if not frames_chunk:
                        break
                    
                    # Process chunk of frames
                    for frame in frames_chunk:
                        try:
                            # Resize frame
                            frame_resized = cv2.resize(frame, (output_width, output_height))
                            
                            # Run YOLO on frame
                            if hasattr(model, 'predict') and hasattr(model, 'draw_detections'):
                                # ONNX model
                                detections = model.predict(frame_resized)
                                processed_frame = model.draw_detections(frame_resized, detections)
                            else:
                                # PyTorch model
                                results = model.predict(frame_resized, verbose=False)
                                processed_frame = results[0].plot()
                            
                            # Write frame
                            out.write(processed_frame)
                            processed_count += 1
                            
                            # Progress update
                            if processed_count % 50 == 0:
                                progress = (frame_count / total_frames) * 100
                                print(f"📊 Progress: {progress:.1f}% ({processed_count}/{total_frames} frames)")
                                
                            # Memory cleanup
                            if processed_count % 100 == 0:
                                gc.collect()
                                
                        except Exception as frame_error:
                            print(f"⚠️ Frame {frame_count} processing failed: {str(frame_error)}")
                            # Write original frame on error
                            out.write(frame_resized)
                    
                    # Clear chunk from memory
                    frames_chunk.clear()
                    gc.collect()
            
            finally:
                cap.release()
                out.release()
                cv2.destroyAllWindows()
                gc.collect()
            
            print(f"✅ Video processing complete:")
            print(f"   - Processed frames: {processed_count}/{total_frames}")
            
            # Read the processed video
            with open(temp_output_path, 'rb') as f:
                processed_video_data = f.read()
            
            # Convert to base64
            original_base64 = base64.b64encode(file_data).decode('utf-8')
            processed_base64 = base64.b64encode(processed_video_data).decode('utf-8')
            
            return original_base64, processed_base64, 'video/mp4'
            
        else:
            # Image processing
            print(f"🖼️ Processing image: {filename}")
            
            # Load image
            image = cv2.imread(temp_input_path)
            if image is None:
                raise Exception(f"Could not load image: {filename}")
            
            # Run inference
            if hasattr(model, 'predict') and hasattr(model, 'draw_detections'):
                # ONNX model
                detections = model.predict(image)
                processed_image = model.draw_detections(image, detections)
            else:
                # PyTorch model
                results = model.predict(image, verbose=False)
                processed_image = results[0].plot()
            
            # Convert to bytes
            _, buffer = cv2.imencode('.jpg', image)
            original_bytes = buffer.tobytes()
            
            _, buffer = cv2.imencode('.jpg', processed_image)
            processed_bytes = buffer.tobytes()
            
            # Convert to base64
            original_base64 = base64.b64encode(original_bytes).decode('utf-8')
            processed_base64 = base64.b64encode(processed_bytes).decode('utf-8')
            
            return original_base64, processed_base64, 'image/jpeg'
    
    finally:
        # Cleanup temp files
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
            os.unlink(temp_output_path)

def process_image_in_memory(file_data, model):
    """Process image entirely in memory using ONNX or PyTorch model"""
    print("🖼️ Processing image in memory...")
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(file_data, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise Exception("Could not decode image")
    
    print(f"📊 Image shape: {image.shape}")
    
    # Check if using ONNX model
    if hasattr(model, 'predict') and hasattr(model, 'draw_detections'):
        # ONNX model
        print("🤖 Using ONNX model for inference...")
        detections = model.predict(image)
        processed_image = model.draw_detections(image, detections)
        print(f"🎯 Found {len(detections)} detections")
    else:
        # PyTorch model (fallback)
        print("🤖 Using PyTorch model for inference...")
        results = model.predict(image, verbose=False, conf=0.3, imgsz=YOLO_PROCESSING_SIZE)
        processed_image = results[0].plot()
    
    # Convert both images to base64
    _, original_buffer = cv2.imencode('.jpg', image)
    _, processed_buffer = cv2.imencode('.jpg', processed_image)
    
    original_base64 = base64.b64encode(original_buffer).decode('utf-8')
    processed_base64 = base64.b64encode(processed_buffer).decode('utf-8')
    
    return original_base64, processed_base64, 'image/jpeg'

def process_video_in_memory(file_data, model):
    """
    Process video entirely in memory using temporary files only during processing
    """
    temp_input = None
    temp_output = None
    
    try:
        print("🎬 Starting in-memory video processing...")
        
        # Check available memory before starting
        import psutil
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        print(f"🧠 Available memory: {available_memory:.1f} MB")
        
        if available_memory < 1000:  # Less than 1GB available
            raise Exception("Insufficient memory available for video processing")
        
        # Create temporary files
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input_file:
            temp_input_file.write(file_data)
            temp_input_path = temp_input_file.name
            temp_input = temp_input_path
        
        # Create output temp file with MP4 extension to avoid codec issues
        temp_output_path = tempfile.mktemp(suffix='.mp4')
        temp_output = temp_output_path
        
        print(f"📁 Created temp input: {temp_input_path}")
        print(f"📁 Created temp output: {temp_output_path}")
        
        # Process the video
        actual_output_path = process_video_with_temp_files(temp_input_path, temp_output_path, model)
        
        # Read the processed video back into memory
        with open(actual_output_path, 'rb') as f:
            processed_video_data = f.read()
        
        # Convert to base64
        original_base64 = base64.b64encode(file_data).decode('utf-8')
        processed_base64 = base64.b64encode(processed_video_data).decode('utf-8')
        
        print(f"✅ Video processing completed successfully")
        print(f"📊 Original size: {len(file_data) / (1024*1024):.2f} MB")
        print(f"📊 Processed size: {len(processed_video_data) / (1024*1024):.2f} MB")
        
        return original_base64, processed_base64, 'video/mp4'
        
    except Exception as e:
        print(f"❌ Error in process_video_in_memory: {str(e)}")
        raise e
    finally:
        # Clean up temporary files
        print("🧹 Cleaning up temporary files...")
        if temp_input and os.path.exists(temp_input):
            os.unlink(temp_input)
            print(f"🧹 Removed temp input: {temp_input}")
        if temp_output and os.path.exists(temp_output):
            os.unlink(temp_output)
            print(f"🧹 Removed temp output: {temp_output}")

def process_video_with_temp_files(input_path, output_path, model):
    """
    Process video with MP4 H.264 output for Cloud Run compatibility
    """
    print(f"🎬 Processing video with MP4 H.264 output...")
    
    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise Exception(f"Could not open video file: {input_path}")
    
    # Get video properties
    original_fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = total_frames / original_fps
    
    # Reasonable size reduction
    max_dimension = 720
    if width > height:
        output_width = max_dimension
        output_height = int(height * (max_dimension / width))
    else:
        output_height = max_dimension
        output_width = int(width * (max_dimension / height))
    
    # Ensure even dimensions
    output_width = output_width - (output_width % 2)
    output_height = output_height - (output_height % 2)
    
    # Keep reasonable FPS
    output_fps = min(30, original_fps)
    
    print(f"📊 Processing plan:")
    print(f"   - Input: {width}x{height} @ {original_fps} fps")
    print(f"   - Output: {output_width}x{output_height} @ {output_fps} fps (WebM)")
    print(f"   - AI will process: ALL frames")
    
    # Force MP4 output for Cloud Run compatibility
    if not output_path.endswith('.mp4'):
        output_path = os.path.splitext(output_path)[0] + '.mp4'
    
    print(f"🌐 Creating MP4 H.264 video writer for Cloud Run...")
    
    # Try H.264 codecs that work well on Cloud Run
    h264_codecs = [
        ('mp4v', 'MPEG-4 (most compatible)'),
        ('XVID', 'XVID (good fallback)'),
        ('MJPG', 'Motion JPEG (always works)')
    ]
    
    out = None
    working_codec = None
    
    for codec_fourcc, codec_name in h264_codecs:
        try:
            print(f"🔧 Testing {codec_name}...")
            fourcc = cv2.VideoWriter_fourcc(*codec_fourcc)
            out = cv2.VideoWriter(output_path, fourcc, output_fps, (output_width, output_height))
            
            if out and out.isOpened():
                # Test write
                test_frame = np.ones((output_height, output_width, 3), dtype=np.uint8) * 128
                success = out.write(test_frame)
                if success:
                    working_codec = codec_fourcc
                    print(f"✅ {codec_name} working on Cloud Run!")
                    break
                else:
                    print(f"❌ {codec_name} write test failed")
                    out.release()
                    out = None
            else:
                print(f"❌ {codec_name} initialization failed")
                if out:
                    out.release()
                out = None
        except Exception as e:
            print(f"❌ {codec_name} error: {e}")
            if out:
                out.release()
            out = None
    
    if not out:
        cap.release()
        raise Exception("Could not create video writer with any codec")
    
    print(f"🎯 Using {working_codec} codec for output: {os.path.basename(output_path)}")
    
    frame_count = 0
    written_count = 0
    ai_processed = 0
    failed_writes = 0
    
    # Skip frames to match target FPS
    frame_skip = max(1, original_fps // output_fps)
    
    print(f"🎬 Processing every {frame_skip} frames with AI on ALL processed frames...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames to match target FPS
            if frame_count % frame_skip != 0:
                continue
            
            try:
                # Resize frame
                frame_resized = cv2.resize(frame, (output_width, output_height))
                
                # Run AI on EVERY processed frame
                try:
                    if hasattr(model, 'predict') and hasattr(model, 'draw_detections'):
                        # ONNX model
                        detections = model.predict(frame_resized)
                        if len(detections) > 0:
                            processed_frame = model.draw_detections(frame_resized, detections)
                            if frame_count % 50 == 0:
                                print(f"🎯 Frame {frame_count}: AI detected {len(detections)} objects")
                        else:
                            processed_frame = frame_resized
                        ai_processed += 1
                    else:
                        # PyTorch model
                        results = model.predict(frame_resized, verbose=False, conf=0.3, imgsz=YOLO_PROCESSING_SIZE)
                        processed_frame = results[0].plot()
                        ai_processed += 1
                    
                    frame_to_write = processed_frame
                except Exception as ai_error:
                    print(f"⚠️ AI failed on frame {frame_count}: {ai_error}")
                    frame_to_write = frame_resized
                    ai_processed += 1
                
                # Ensure proper frame format
                if frame_to_write.dtype != np.uint8:
                    frame_to_write = frame_to_write.astype(np.uint8)
                
                # Write frame
                write_success = out.write(frame_to_write)
                
                if write_success:
                    written_count += 1
                else:
                    failed_writes += 1
                    if failed_writes % 50 == 0:
                        print(f"⚠️ Write failures: {failed_writes}")
                
                # Progress reporting
                if written_count % 50 == 0 and written_count > 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"📹 Progress: {progress:.1f}% - Written: {written_count}, AI: {ai_processed}")
                
                # Memory cleanup
                if frame_count % 25 == 0:
                    gc.collect()
                
            except Exception as frame_error:
                print(f"❌ Error processing frame {frame_count}: {frame_error}")
                failed_writes += 1
                continue
        
        # Calculate final duration
        final_duration = written_count / output_fps if output_fps > 0 else 0
        
        print(f"✅ Video processing completed!")
        print(f"   - Input frames: {frame_count}")
        print(f"   - Output frames: {written_count}")
        print(f"   - AI processed: {ai_processed}")
        print(f"   - Failed writes: {failed_writes}")
        print(f"   - Final duration: {final_duration:.1f} seconds")
        print(f"   - Output format: {os.path.splitext(output_path)[1]}")
        
    finally:
        print("🧹 Cleanup...")
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        gc.collect()
    
    # Verify output
    if not os.path.exists(output_path):
        raise Exception("Output file not created")
    
    file_size = os.path.getsize(output_path)
    if file_size < 1024:
        raise Exception("Output file too small")
    
    if written_count == 0:
        raise Exception("No frames were successfully written to output video")
    
    print(f"📊 Final output: {file_size / (1024*1024):.2f} MB")
    
    return output_path

def process_video_with_ffmpeg_fallback(input_path, output_path, model):
    """
    Process video with FFmpeg fallback when OpenCV fails
    """
    print(f"🎬 Attempting video processing with FFmpeg fallback...")
    
    # First try OpenCV approach
    try:
        return process_video_with_temp_files(input_path, output_path, model)
    except Exception as opencv_error:
        print(f"❌ OpenCV failed: {opencv_error}")
        print(f"🔄 Falling back to FFmpeg approach...")
        
        # FFmpeg approach: process frames and use FFmpeg to encode
        return process_video_with_ffmpeg(input_path, output_path, model)

def process_video_with_ffmpeg(input_path, output_path, model):
    """
    Process video using frame extraction + FFmpeg encoding
    """
    print(f"🛠️ Processing video with FFmpeg...")
    
    # Create temp directory for frames
    with tempfile.TemporaryDirectory() as temp_dir:
        frames_dir = os.path.join(temp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        # Extract frames using OpenCV
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {input_path}")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Target resolution
        max_dimension = 720
        if width > height:
            output_width = max_dimension
            output_height = int(height * (max_dimension / width))
        else:
            output_height = max_dimension
            output_width = int(width * (max_dimension / height))
        
        # Ensure even dimensions for FFmpeg
        output_width = output_width - (output_width % 2)
        output_height = output_height - (output_height % 2)
        
        output_fps = min(30, fps)
        frame_skip = max(1, fps // output_fps)
        
        print(f"📊 FFmpeg processing plan:")
        print(f"   - Input: {width}x{height} @ {fps} fps")
        print(f"   - Output: {output_width}x{output_height} @ {output_fps} fps")
        print(f"   - Frame skip: every {frame_skip} frames")
        
        frame_count = 0
        saved_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Skip frames to match target FPS
                if frame_count % frame_skip != 0:
                    continue
                
                # Resize frame
                frame_resized = cv2.resize(frame, (output_width, output_height))
                
                # Run AI on this frame
                try:
                    if hasattr(model, 'predict') and hasattr(model, 'draw_detections'):
                        # ONNX model
                        detections = model.predict(frame_resized)
                        if len(detections) > 0:
                            processed_frame = model.draw_detections(frame_resized, detections)
                        else:
                            processed_frame = frame_resized
                    else:
                        # PyTorch model
                        results = model.predict(frame_resized, verbose=False, conf=0.3, imgsz=YOLO_PROCESSING_SIZE)
                        processed_frame = results[0].plot()
                    
                    # Save frame as image
                    frame_filename = os.path.join(frames_dir, f"frame_{saved_frames:06d}.jpg")
                    cv2.imwrite(frame_filename, processed_frame)
                    saved_frames += 1
                    
                    if saved_frames % 50 == 0:
                        print(f"📹 Processed {saved_frames} frames...")
                        
                except Exception as ai_error:
                    print(f"⚠️ AI failed on frame {frame_count}: {ai_error}")
                    # Save original frame
                    frame_filename = os.path.join(frames_dir, f"frame_{saved_frames:06d}.jpg")
                    cv2.imwrite(frame_filename, frame_resized)
                    saved_frames += 1
        
        finally:
            cap.release()
        
        print(f"✅ Extracted and processed {saved_frames} frames")
        
        # Use FFmpeg to create video from frames
        if saved_frames == 0:
            raise Exception("No frames were processed")
        
        # Ensure output is MP4
        if not output_path.endswith('.mp4'):
            output_path = os.path.splitext(output_path)[0] + '.mp4'
        
        print(f"🎬 Creating MP4 video with FFmpeg...")
        
        # FFmpeg command to create MP4 from frames
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # -y to overwrite output file
            '-framerate', str(output_fps),
            '-i', os.path.join(frames_dir, 'frame_%06d.jpg'),
            '-c:v', 'libx264',  # H.264 codec
            '-preset', 'fast',  # Fast encoding
            '-crf', '28',  # Good quality/size balance
            '-pix_fmt', 'yuv420p',  # Web compatibility
            output_path
        ]
        
        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print(f"✅ FFmpeg video creation successful!")
            else:
                print(f"❌ FFmpeg error: {result.stderr}")
                raise Exception(f"FFmpeg failed: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            raise Exception("FFmpeg encoding timed out")
        except FileNotFoundError:
            raise Exception("FFmpeg not found - not available on Cloud Run")
        
        # Verify output
        if not os.path.exists(output_path):
            raise Exception("FFmpeg did not create output file")
        
        file_size = os.path.getsize(output_path)
        if file_size < 1024:
            raise Exception("Output file too small")
        
        print(f"📊 FFmpeg output: {file_size / (1024*1024):.2f} MB")
        
        return output_path

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
            'cookiefile': os.path.join(os.path.dirname(__file__), 'cookies.txt'),
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
