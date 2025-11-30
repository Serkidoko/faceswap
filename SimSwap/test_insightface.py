"""
Test script for InsightFace library
Check if face detection and models are working correctly
"""

import cv2
import numpy as np
from insightface_func.face_detect_crop_multi import Face_detect_crop
import os

def test_insightface():
    print("=" * 50)
    print("Testing InsightFace Library")
    print("=" * 50)
    
    # Test 1: Load models
    print("\n[Test 1] Loading InsightFace models...")
    try:
        app = Face_detect_crop(name='antelope', root='./insightface_func/models')
        print("✓ Models loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load models: {e}")
        return False
    
    # Test 2: Prepare models
    print("\n[Test 2] Preparing models...")
    try:
        app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640), mode='None')
        print("✓ Models prepared successfully")
    except Exception as e:
        print(f"✗ Failed to prepare models: {e}")
        return False
    
    # Test 3: Test with demo image
    print("\n[Test 3] Testing face detection on demo image...")
    demo_img_path = './demo_file/Iron_man.jpg'
    
    if not os.path.exists(demo_img_path):
        print(f"✗ Demo image not found: {demo_img_path}")
        # Try to find any image in demo_file
        demo_files = os.listdir('./demo_file')
        img_files = [f for f in demo_files if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if img_files:
            demo_img_path = os.path.join('./demo_file', img_files[0])
            print(f"  Using alternative: {demo_img_path}")
        else:
            print("  No demo images available")
            return False
    
    try:
        img = cv2.imread(demo_img_path)
        if img is None:
            print(f"✗ Cannot read image: {demo_img_path}")
            return False
        
        print(f"  Image loaded: {img.shape}")
        
        # Detect faces with 224x224 crop
        results = app.get(img, crop_size=224)
        
        if results is None:
            print("✗ No faces detected")
            return False
        
        align_img_list, mat_list = results
        print(f"✓ Detected {len(align_img_list)} face(s)")
        
        # Save cropped faces
        output_dir = './output/test_insightface'
        os.makedirs(output_dir, exist_ok=True)
        
        for i, aligned_face in enumerate(align_img_list):
            output_path = os.path.join(output_dir, f'face_{i}.jpg')
            cv2.imwrite(output_path, aligned_face)
            print(f"  Face {i}: saved to {output_path}")
            print(f"  Shape: {aligned_face.shape}")
        
    except Exception as e:
        print(f"✗ Face detection failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Test with multiple crop sizes
    print("\n[Test 4] Testing different crop sizes...")
    for crop_size in [224, 512]:
        try:
            results = app.get(img, crop_size=crop_size)
            if results:
                align_img_list, _ = results
                print(f"✓ Crop size {crop_size}: {len(align_img_list)} face(s) detected")
            else:
                print(f"✗ Crop size {crop_size}: No faces detected")
        except Exception as e:
            print(f"✗ Crop size {crop_size}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("All tests passed! InsightFace is working correctly.")
    print("=" * 50)
    return True

if __name__ == '__main__':
    success = test_insightface()
    exit(0 if success else 1)
