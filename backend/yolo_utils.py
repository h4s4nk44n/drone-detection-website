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
import yaml
import re
import time
import psutil  # Add psutil for memory monitoring

YOLO_PROCESSING_SIZE = 832  # High accuracy processing for all media

def cleanup_memory():
    """
    Force garbage collection and memory cleanup
    """
    import gc
    gc.collect()
    
def log_memory_usage(stage=""):
    """
    Log current memory usage for monitoring
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / (1024 * 1024)
        available_memory = psutil.virtual_memory().available / (1024 * 1024)
        print(f"🧠 Memory {stage}: {memory_mb:.1f}MB used, {available_memory:.1f}MB available")
        
        # Warning if memory usage is high
        if memory_mb > 3000:  # Warning at 3GB usage
            print(f"⚠️ High memory usage detected: {memory_mb:.1f}MB")
        
        return memory_mb
    except Exception as e:
        print(f"⚠️ Memory monitoring failed: {e}")
        return 0

def cleanup_opencv_resources():
    """
    Clean up OpenCV resources and destroy windows
    """
    try:
        cv2.destroyAllWindows()
        gc.collect()
    except Exception as e:
        print(f"⚠️ OpenCV cleanup warning: {e}")

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

def process_file_in_memory(file_data, file_ext, filename, model, extract_training_data=False, confidence_threshold=0.3):
    """
    Process file in memory with proper MIME types for web compatibility
    Enhanced with FFmpeg fallback for reliable video processing
    Optionally extract training data with YOLO format labels
    Now includes memory management for better multi-user support
    """
    print(f"🔄 Processing file in memory: {filename}")
    log_memory_usage("start")
    
    if extract_training_data:
        print(f"🎯 Training data extraction enabled with confidence threshold: {confidence_threshold}")
        print(f"🔧 Strategy: model.predict() uses conf=0.1 to capture more detections")
        print(f"🔧 Then filter detections by user threshold: {confidence_threshold}")
        print(f"🔧 This ensures we see all detections and apply user's confidence filter correctly")
    
    # Training data extraction setup
    training_images = []
    training_labels = []
    extracted_count = 0
    
    # Create temporary files
    with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as temp_input:
        temp_input.write(file_data)
        temp_input_path = temp_input.name
    
    # Save original file data for base64 encoding before deleting for memory management
    original_file_data = file_data
    
    # Clear file_data from memory immediately after writing to temp file
    del file_data
    cleanup_memory()
    log_memory_usage("after temp file creation")
    
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
            cap.release()  # Release early to free memory
            
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
            
            # Try OpenCV approach first, fallback to FFmpeg
            processed_successfully = False
            opencv_error = None
            processed_count = 0  # Initialize to avoid scope issues
            
            try:
                print("🔄 Attempting OpenCV video processing...")
                
                # Initialize video writer with H.264 codec
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(temp_output_path, fourcc, target_fps, (output_width, output_height))
                
                if not out.isOpened():
                    raise Exception("OpenCV VideoWriter failed to initialize")
                
                # Test write to verify codec works
                test_frame = np.ones((output_height, output_width, 3), dtype=np.uint8) * 128
                success = out.write(test_frame)
                if not success:
                    out.release()
                    raise Exception("OpenCV VideoWriter test write failed")
                
                # Process video with OpenCV
                cap = cv2.VideoCapture(temp_input_path)
                frame_count = 0
                processed_count = 0  # Ensure it's in this scope
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
                                
                                # Run AI on frame
                                if hasattr(model, 'predict') and hasattr(model, 'draw_detections'):
                                    # ONNX model
                                    detections = model.predict(frame_resized)
                                    processed_frame = model.draw_detections(frame_resized, detections)
                                    
                                    # Extract training data for ONNX model if needed
                                    if extract_training_data:
                                        # Find all detections that meet confidence threshold
                                        valid_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
                                        
                                        if valid_detections:
                                            # Use original resolution frame for training data
                                            original_frame = cv2.resize(frame, (width, height))
                                            training_images.append(original_frame.copy())
                                            
                                            # Create YOLO format labels for ALL detections in this frame
                                            frame_labels = []
                                            for detection in valid_detections:
                                                x1, y1, x2, y2 = detection['bbox']
                                                center_x = (x1 + x2) / 2 / width
                                                center_y = (y1 + y2) / 2 / height
                                                bbox_width = (x2 - x1) / width
                                                bbox_height = (y2 - y1) / height
                                                
                                                # Assuming class 0 for drone (adjust based on your model)
                                                label = f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                                                frame_labels.append(label)
                                            
                                            # Join all labels for this frame with newlines
                                            combined_labels = "\n".join(frame_labels)
                                            training_labels.append(combined_labels)
                                            extracted_count += len(valid_detections)  # Count all detections
                                            
                                            if frame_count % 100 == 0:  # Log multi-detection frames
                                                print(f"🎯 ONNX Frame {frame_count}: Extracted {len(valid_detections)} drones for training")
                                else:
                                    # PyTorch model
                                    if extract_training_data:
                                        # For training data extraction, use lower model conf to capture more detections
                                        results = model.predict(frame_resized, verbose=False, conf=0.1)
                                    else:
                                        # For regular processing, use standard conf
                                        results = model.predict(frame_resized, verbose=False, conf=0.3)
                                    
                                    processed_frame = results[0].plot()
                                    
                                    # Extract training data for PyTorch model if needed
                                    if extract_training_data:
                                        boxes = results[0].boxes
                                        print(f"🔍 Frame {frame_count}: model.predict() with conf=0.1 found {len(boxes) if boxes is not None else 0} detections")
                                        
                                        if boxes is not None and len(boxes) > 0:
                                            # Show all detection confidences
                                            all_confidences = boxes.conf.cpu().numpy()
                                            print(f"🔍 Frame {frame_count}: Detection confidences: {[f'{conf:.3f}' for conf in all_confidences]}")
                                            
                                            # Filter by user's confidence threshold
                                            confident_boxes = boxes[boxes.conf >= confidence_threshold]
                                            print(f"🔍 Frame {frame_count}: {len(confident_boxes)} detections above user threshold {confidence_threshold}")
                                            
                                            if len(confident_boxes) > 0:
                                                # Use original resolution frame for training data
                                                original_frame = cv2.resize(frame, (width, height))
                                                training_images.append(original_frame.copy())
                                                
                                                # Extract ALL detections that meet confidence threshold
                                                frame_labels = []
                                                for box in confident_boxes:
                                                    conf = float(box.conf.cpu().numpy())
                                                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                                    img_height, img_width = frame.shape[:2]
                                                    center_x = (x1 + x2) / 2 / img_width
                                                    center_y = (y1 + y2) / 2 / img_height
                                                    bbox_width = (x2 - x1) / img_width
                                                    bbox_height = (y2 - y1) / img_height
                                                    class_id = int(box.cls.cpu().numpy())
                                                    
                                                    label = f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                                                    frame_labels.append(label)
                                                
                                                # Join all labels for this frame with newlines
                                                combined_labels = "\n".join(frame_labels)
                                                training_labels.append(combined_labels)
                                                extracted_count += len(confident_boxes)  # Count all detections
                                                
                                                print(f"🎯 Frame {frame_count}: Extracted {len(confident_boxes)} drones for training (confidences: {[f'{float(box.conf.cpu().numpy()):.3f}' for box in confident_boxes]})")
                                            else:
                                                print(f"⚠️ Frame {frame_count}: All detections below threshold {confidence_threshold}")
                                        else:
                                            print(f"⚠️ Frame {frame_count}: No detections found by model")
                                
                                # Write frame
                                write_success = out.write(processed_frame)
                                if not write_success:
                                    raise Exception("Frame write failed")
                                processed_count += 1
                                
                                # Progress update with memory monitoring
                                if processed_count % 50 == 0:
                                    progress = (frame_count / total_frames) * 100
                                    extraction_info = f", extracted: {extracted_count}" if extract_training_data else ""
                                    print(f"📊 OpenCV Progress: {progress:.1f}% ({processed_count}/{total_frames} frames{extraction_info})")
                                    log_memory_usage(f"frame {processed_count}")
                                    
                                # Memory cleanup every 100 frames
                                if processed_count % 100 == 0:
                                    cleanup_memory()
                                    
                            except Exception as frame_error:
                                print(f"⚠️ Frame {frame_count} processing failed: {str(frame_error)}")
                                # Write original frame on error
                                out.write(frame_resized)
                        
                        # Clear chunk from memory and force cleanup
                        frames_chunk.clear()
                        del frames_chunk
                        cleanup_memory()
                
                finally:
                    cap.release()
                    out.release()
                    cleanup_opencv_resources()
                    cleanup_memory()
                    log_memory_usage("after OpenCV processing")
                
                # Verify OpenCV output
                if os.path.exists(temp_output_path) and os.path.getsize(temp_output_path) > 1024:
                    print(f"✅ OpenCV video processing complete: {processed_count}/{total_frames} frames")
                    processed_successfully = True
                else:
                    raise Exception("OpenCV output file invalid")
                    
            except Exception as opencv_err:
                opencv_error = opencv_err
                print(f"❌ OpenCV processing failed: {str(opencv_error)}")
                print("🔄 Falling back to FFmpeg approach...")
                
                # Clean up failed OpenCV output
                if os.path.exists(temp_output_path):
                    os.unlink(temp_output_path)
                
                # Use FFmpeg fallback - now supports training data extraction
                try:
                    ffmpeg_result = process_video_with_ffmpeg(temp_input_path, temp_output_path, model, extract_training_data, confidence_threshold)
                    
                    if extract_training_data:
                        # FFmpeg now returns training data
                        if len(ffmpeg_result) == 4:  # processed_output, training_images, training_labels, extracted_count
                            processed_output, ffmpeg_training_images, ffmpeg_training_labels, ffmpeg_extracted_count = ffmpeg_result
                            training_images.extend(ffmpeg_training_images)
                            training_labels.extend(ffmpeg_training_labels)
                            extracted_count += ffmpeg_extracted_count
                            print(f"🎯 FFmpeg extracted {ffmpeg_extracted_count} training samples")
                        else:
                            processed_output = ffmpeg_result
                    else:
                        processed_output = ffmpeg_result
                    
                    if os.path.exists(processed_output) and os.path.getsize(processed_output) > 1024:
                        temp_output_path = processed_output
                        processed_successfully = True
                        print("✅ FFmpeg video processing complete")
                    else:
                        raise Exception("FFmpeg output file invalid")
                        
                except Exception as ffmpeg_error:
                    print(f"❌ FFmpeg processing also failed: {str(ffmpeg_error)}")
                    raise Exception(f"Both OpenCV and FFmpeg failed. OpenCV: {opencv_error}, FFmpeg: {ffmpeg_error}")
            
            if not processed_successfully:
                raise Exception("Video processing failed with all methods")
            
            # Read the processed video
            with open(temp_output_path, 'rb') as f:
                processed_video_data = f.read()
            
            # Convert to base64
            original_base64 = base64.b64encode(original_file_data).decode('utf-8')
            processed_base64 = base64.b64encode(processed_video_data).decode('utf-8')
            
            # Clear original file data from memory after base64 conversion
            del original_file_data
            
            print(f"✅ Video processing complete:")
            print(f"   - Method: {'OpenCV' if opencv_error is None else 'FFmpeg'}")
            print(f"   - Output size: {len(processed_video_data) / (1024*1024):.2f}MB")
            if extract_training_data:
                print(f"🎯 Training Data Extraction Summary:")
                print(f"   - Training samples extracted: {extracted_count}")
                print(f"   - User confidence threshold: {confidence_threshold}")
                # Show frame statistics if available
                if processed_count > 0:
                    print(f"   - Total frames processed: {processed_count}")
                    print(f"   - Extraction rate: {extracted_count/processed_count*100:.1f}% of processed frames")
                    print(f"   - Note: Each frame was checked with model conf=0.1, then filtered by user threshold")
                elif opencv_error is not None:
                    print(f"   - Note: Frame statistics not available with FFmpeg fallback")
                else:
                    print(f"   - Note: No frames were processed")
            
            # Clear processed video data from memory after base64 conversion
            del processed_video_data
            cleanup_memory()
            log_memory_usage("after base64 conversion")
            
            # Return with or without training data
            if extract_training_data and extracted_count > 0:
                zip_data = create_training_dataset_zip(training_images, training_labels, filename)
                # Clear training data from memory after ZIP creation
                training_images.clear()
                training_labels.clear()
                cleanup_memory()
                log_memory_usage("after ZIP creation")
                return original_base64, processed_base64, 'video/mp4', zip_data, extracted_count
            elif extract_training_data and extracted_count == 0:
                # Provide better error messages now that both OpenCV and FFmpeg support training data extraction
                print(f"⚠️ No training data extracted - no detections met confidence threshold {confidence_threshold}")
                print(f"💡 Suggestion: Try lowering confidence threshold to 0.1 or 0.2")
                if opencv_error is not None:
                    print(f"   - Note: Used FFmpeg fallback due to OpenCV error: {str(opencv_error)[:100]}...")
                # Clear empty training data from memory
                training_images.clear()
                training_labels.clear()
                cleanup_memory()
                return original_base64, processed_base64, 'video/mp4'
            else:
                cleanup_memory()
                log_memory_usage("final cleanup")
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
                
                # Extract training data for ONNX model if needed
                if extract_training_data:
                    for detection in detections:
                        if detection['confidence'] >= confidence_threshold:
                            training_images.append(image.copy())
                            
                            # Create YOLO format label
                            x1, y1, x2, y2 = detection['bbox']
                            img_height, img_width = image.shape[:2]
                            center_x = (x1 + x2) / 2 / img_width
                            center_y = (y1 + y2) / 2 / img_height
                            bbox_width = (x2 - x1) / img_width
                            bbox_height = (y2 - y1) / img_height
                            
                            label = f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                            training_labels.append(label)
                            extracted_count += 1
            else:
                # PyTorch model
                if extract_training_data:
                    # For training data extraction, use lower model conf to capture more detections
                    results = model.predict(image, verbose=False, conf=0.1)
                else:
                    # For regular processing, use standard conf
                    results = model.predict(image, verbose=False, conf=0.3)
                    
                processed_image = results[0].plot()
                
                # Extract training data for PyTorch model if needed
                if extract_training_data:
                    boxes = results[0].boxes
                    print(f"🔍 Image: model.predict() with conf=0.1 found {len(boxes) if boxes is not None else 0} detections")
                    
                    if boxes is not None and len(boxes) > 0:
                        # Show all detection confidences
                        all_confidences = boxes.conf.cpu().numpy()
                        print(f"🔍 Image: Detection confidences: {[f'{conf:.3f}' for conf in all_confidences]}")
                        
                        # Filter by user's confidence threshold
                        confident_boxes = boxes[boxes.conf >= confidence_threshold]
                        print(f"🔍 Image: {len(confident_boxes)} detections above user threshold {confidence_threshold}")
                        
                        if len(confident_boxes) > 0:
                            training_images.append(image.copy())
                            
                            # Extract ALL detections that meet confidence threshold
                            frame_labels = []
                            for box in confident_boxes:
                                conf = float(box.conf.cpu().numpy())
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                img_height, img_width = image.shape[:2]
                                center_x = (x1 + x2) / 2 / img_width
                                center_y = (y1 + y2) / 2 / img_height
                                bbox_width = (x2 - x1) / img_width
                                bbox_height = (y2 - y1) / img_height
                                class_id = int(box.cls.cpu().numpy())
                                
                                label = f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                                frame_labels.append(label)
                            
                            # Join all labels for this frame with newlines
                            combined_labels = "\n".join(frame_labels)
                            training_labels.append(combined_labels)
                            extracted_count += len(confident_boxes)  # Count all detections
                            
                            print(f"🎯 Image: Extracted {len(confident_boxes)} drones for training (confidences: {[f'{float(box.conf.cpu().numpy()):.3f}' for box in confident_boxes]})")
                        else:
                            print(f"⚠️ Image: All detections below threshold {confidence_threshold}")
                    else:
                        print(f"⚠️ Image: No detections found by model")
            
            # Convert to bytes
            _, buffer = cv2.imencode('.jpg', image)
            original_bytes = buffer.tobytes()
            
            _, buffer = cv2.imencode('.jpg', processed_image)
            processed_bytes = buffer.tobytes()
            
            # Convert to base64
            original_base64 = base64.b64encode(original_bytes).decode('utf-8')
            processed_base64 = base64.b64encode(processed_bytes).decode('utf-8')
            
            print(f"✅ Image processing complete")
            if extract_training_data:
                print(f"   - Training samples extracted: {extracted_count}")
                if extracted_count == 0:
                    print(f"⚠️ No training data extracted from image - all detections were below confidence threshold {confidence_threshold}")
            
            # Return with or without training data
            if extract_training_data and extracted_count > 0:
                zip_data = create_training_dataset_zip(training_images, training_labels, filename)
                return original_base64, processed_base64, 'image/jpeg', zip_data, extracted_count
            elif extract_training_data and extracted_count == 0:
                print(f"⚠️ No training data extracted from image - all detections were below confidence threshold {confidence_threshold}")
                return original_base64, processed_base64, 'image/jpeg'
            else:
                return original_base64, processed_base64, 'image/jpeg'
    
    finally:
        # Cleanup temp files
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if 'temp_output_path' in locals() and os.path.exists(temp_output_path):
            os.unlink(temp_output_path)

def create_training_dataset_zip(training_images, training_labels, original_filename):
    """
    Create a ZIP file containing training images and YOLO format labels
    """
    import zipfile
    from io import BytesIO
    import yaml
    import re
    
    print(f"📦 Creating training dataset ZIP with {len(training_images)} samples...")
    
    # Create a safe filename prefix from the original filename
    # Remove file extension and replace spaces/special chars with underscores
    base_name = os.path.splitext(original_filename)[0]
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', base_name)
    safe_name = re.sub(r'_+', '_', safe_name)  # Replace multiple underscores with single
    safe_name = safe_name.strip('_')  # Remove leading/trailing underscores
    
    if not safe_name:  # Fallback if filename becomes empty
        safe_name = "drone_dataset"
    
    print(f"📋 Using base name: '{safe_name}' (from '{original_filename}')")
    
    # Create in-memory ZIP file
    zip_buffer = BytesIO()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Create images and labels directories
        for i, (image, label) in enumerate(zip(training_images, training_labels)):
            # Create unique filenames using the original filename
            image_filename = f"images/{safe_name}_{i:06d}.jpg"
            label_filename = f"labels/{safe_name}_{i:06d}.txt"
            
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
        readme_content = f"""# Drone Detection Training Dataset

This dataset was automatically extracted from: {original_filename}

## Contents:
- images/: Training images with drone detections ({len(training_images)} files)
- labels/: YOLO format annotation files ({len(training_labels)} files)
- dataset.yaml: Dataset configuration for YOLO training

## Dataset Statistics:
- Total samples: {len(training_images)}
- Image format: JPG
- Label format: YOLO (class_id center_x center_y width height - normalized coordinates)
- Filename pattern: {safe_name}_XXXXXX.jpg/.txt

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

Generated by Drone Detection System
Extracted on: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        zip_file.writestr('README.md', readme_content)
    
    zip_data = zip_buffer.getvalue()
    zip_buffer.close()
    
    print(f"✅ Training dataset ZIP created: {len(zip_data) / (1024*1024):.2f}MB")
    print(f"📋 Files naming pattern: {safe_name}_XXXXXX.jpg/.txt")
    return zip_data

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
        results = model.predict(image, verbose=False)
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

def process_video_with_ffmpeg(input_path, output_path, model, extract_training_data=False, confidence_threshold=0.3):
    """
    Process video using frame extraction + FFmpeg encoding
    Now supports training data extraction
    """
    print(f"🛠️ Processing video with FFmpeg...")
    if extract_training_data:
        print(f"🎯 FFmpeg processing with training data extraction enabled (confidence: {confidence_threshold})")
    
    # Training data collection
    training_images = []
    training_labels = []
    extracted_count = 0
    
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
                            
                            # Extract training data for ONNX model if needed
                            if extract_training_data:
                                # Convert ONNX detections to training data format
                                # Find all detections that meet confidence threshold
                                valid_detections = [d for d in detections if d['confidence'] >= confidence_threshold]
                                
                                if valid_detections:
                                    # Use original resolution frame for training data
                                    original_frame = cv2.resize(frame, (width, height))
                                    training_images.append(original_frame.copy())
                                    
                                    # Create YOLO format labels for ALL detections in this frame
                                    frame_labels = []
                                    for detection in valid_detections:
                                        x1, y1, x2, y2 = detection['bbox']
                                        center_x = (x1 + x2) / 2 / width
                                        center_y = (y1 + y2) / 2 / height
                                        bbox_width = (x2 - x1) / width
                                        bbox_height = (y2 - y1) / height
                                        
                                        # Assuming class 0 for drone (adjust based on your model)
                                        label = f"0 {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                                        frame_labels.append(label)
                                    
                                    # Join all labels for this frame with newlines
                                    combined_labels = "\n".join(frame_labels)
                                    training_labels.append(combined_labels)
                                    extracted_count += len(valid_detections)  # Count all detections
                                    
                                    if frame_count % 100 == 0:  # Log multi-detection frames
                                        print(f"🎯 ONNX Frame {frame_count}: Extracted {len(valid_detections)} drones for training")
                        else:
                            processed_frame = frame_resized
                    else:
                        # PyTorch model
                        if extract_training_data:
                            # For training data extraction, use lower model conf to capture more detections
                            results = model.predict(frame_resized, verbose=False, conf=0.1)
                        else:
                            # For regular processing, use standard conf
                            results = model.predict(frame_resized, verbose=False, conf=0.3)
                            
                        processed_frame = results[0].plot()
                        
                        # Extract training data for PyTorch model if needed
                        if extract_training_data:
                            boxes = results[0].boxes
                            print(f"🔍 Frame {frame_count}: model.predict() with conf=0.1 found {len(boxes) if boxes is not None else 0} detections")
                            
                            if boxes is not None and len(boxes) > 0:
                                # Show all detection confidences
                                all_confidences = boxes.conf.cpu().numpy()
                                print(f"🔍 Frame {frame_count}: Detection confidences: {[f'{conf:.3f}' for conf in all_confidences]}")
                                
                                # Filter by user's confidence threshold
                                confident_boxes = boxes[boxes.conf >= confidence_threshold]
                                print(f"🔍 Frame {frame_count}: {len(confident_boxes)} detections above user threshold {confidence_threshold}")
                                
                                if len(confident_boxes) > 0:
                                    # Use original resolution frame for training data
                                    original_frame = cv2.resize(frame, (width, height))
                                    training_images.append(original_frame.copy())
                                    
                                    # Extract ALL detections that meet confidence threshold
                                    frame_labels = []
                                    for box in confident_boxes:
                                        conf = float(box.conf.cpu().numpy())
                                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                        img_height, img_width = frame.shape[:2]
                                        center_x = (x1 + x2) / 2 / img_width
                                        center_y = (y1 + y2) / 2 / img_height
                                        bbox_width = (x2 - x1) / img_width
                                        bbox_height = (y2 - y1) / img_height
                                        class_id = int(box.cls.cpu().numpy())
                                        
                                        label = f"{class_id} {center_x:.6f} {center_y:.6f} {bbox_width:.6f} {bbox_height:.6f}"
                                        frame_labels.append(label)
                                    
                                    # Join all labels for this frame with newlines
                                    combined_labels = "\n".join(frame_labels)
                                    training_labels.append(combined_labels)
                                    extracted_count += len(confident_boxes)  # Count all detections
                                    
                                    print(f"🎯 Frame {frame_count}: Extracted {len(confident_boxes)} drones for training (confidences: {[f'{float(box.conf.cpu().numpy()):.3f}' for box in confident_boxes]})")
                                else:
                                    print(f"⚠️ Frame {frame_count}: All detections below threshold {confidence_threshold}")
                            else:
                                print(f"⚠️ Frame {frame_count}: No detections found by model")
                    
                    frame_to_write = processed_frame
                except Exception as ai_error:
                    print(f"⚠️ AI failed on frame {frame_count}: {ai_error}")
                    frame_to_write = frame_resized
                
                # Save frame as image
                frame_filename = os.path.join(frames_dir, f"frame_{saved_frames:06d}.jpg")
                cv2.imwrite(frame_filename, frame_to_write)
                saved_frames += 1
                
                if saved_frames % 50 == 0:
                    extraction_info = f", extracted: {extracted_count}" if extract_training_data else ""
                    print(f"📹 FFmpeg processed {saved_frames} frames{extraction_info}...")
                        
        finally:
            cap.release()
        
        print(f"✅ FFmpeg extracted and processed {saved_frames} frames")
        if extract_training_data:
            print(f"🎯 FFmpeg training data extraction: {extracted_count} samples collected")
        
        # Clean up memory after frame processing
        cleanup_memory()
        log_memory_usage("after FFmpeg frame processing")
        
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
        
        # Return training data if extracted
        if extract_training_data and extracted_count > 0:
            return output_path, training_images, training_labels, extracted_count
        else:
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
