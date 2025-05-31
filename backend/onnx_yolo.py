import onnxruntime as ort
import numpy as np
import cv2
from typing import List, Tuple
import time

class ONNXYOLODetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.3):
        """Initialize ONNX YOLO detector"""
        print(f"🤖 Loading ONNX model from: {model_path}")
        
        # Configure ONNX Runtime for CPU optimization
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 1  # Optimize for Cloud Run
        
        self.session = ort.InferenceSession(
            model_path, 
            providers=providers,
            sess_options=sess_options
        )
        
        self.conf_threshold = conf_threshold
        
        # Get model input details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        
        print(f"✅ ONNX model loaded successfully")
        print(f"📊 Input shape: {self.input_shape}")
        
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for ONNX model"""
        # Resize image to model input size
        resized = cv2.resize(image, (self.input_width, self.input_height))
        
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb_image.astype(np.float32) / 255.0
        
        # Convert to CHW format and add batch dimension
        input_tensor = np.transpose(normalized, (2, 0, 1))  # HWC to CHW
        input_tensor = np.expand_dims(input_tensor, axis=0)  # Add batch dimension
        
        return input_tensor
    
    def postprocess_detections(self, outputs: np.ndarray, original_shape: Tuple[int, int]) -> List[dict]:
        """Post-process ONNX model outputs"""
        detections = []
        
        # YOLO output format: [batch, num_detections, 85]
        predictions = outputs[0][0]  # Remove batch dimension
        
        orig_height, orig_width = original_shape
        scale_x = orig_width / self.input_width
        scale_y = orig_height / self.input_height
        
        for detection in predictions:
            # Extract bbox, confidence, and class scores
            x_center, y_center, width, height = detection[:4]
            confidence = detection[4]
            
            # Filter by confidence
            if confidence < self.conf_threshold:
                continue
            
            # Convert to corner coordinates and scale to original image
            x1 = int((x_center - width / 2) * scale_x)
            y1 = int((y_center - height / 2) * scale_y)
            x2 = int((x_center + width / 2) * scale_x)
            y2 = int((y_center + height / 2) * scale_y)
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, orig_width))
            y1 = max(0, min(y1, orig_height))
            x2 = max(0, min(x2, orig_width))
            y2 = max(0, min(y2, orig_height))
            
            detections.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': float(confidence),
                'class_name': 'drone'
            })
        
        return detections
    
    def predict(self, image: np.ndarray) -> List[dict]:
        """Run inference on image"""
        start_time = time.time()
        
        # Preprocess
        input_tensor = self.preprocess_image(image)
        
        # Run inference
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        # Post-process
        detections = self.postprocess_detections(outputs, image.shape[:2])
        
        inference_time = time.time() - start_time
        print(f"⚡ ONNX inference time: {inference_time:.3f}s")
        
        return detections
    
    def draw_detections(self, image: np.ndarray, detections: List[dict]) -> np.ndarray:
        """Draw detection boxes on image"""
        result_image = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(result_image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(result_image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        return result_image