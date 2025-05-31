from ultralytics import YOLO
import torch

def convert_yolo_to_onnx():
    """Convert YOLO model to ONNX format"""
    
    print("🔄 Loading YOLO model...")
    model = YOLO('models/best.pt')
    
    print("🔄 Converting to ONNX format...")
    # Export to ONNX with optimizations
    model.export(
        format='onnx',
        imgsz=832,  # Your training size
        optimize=True,  # Optimize for inference
        simplify=True,  # Simplify the model
        dynamic=False,  # Fixed input size for better optimization
        opset=11  # ONNX opset version
    )
    
    print("✅ Model converted successfully!")
    print("📁 ONNX model saved as: models/best.onnx")

if __name__ == "__main__":
    convert_yolo_to_onnx()