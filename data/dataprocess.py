"""
Face Alignment Tool for SimSwap Fine-tuning
==========================================

This script aligns and crops faces from images to prepare training data for SimSwap.

Usage:
    python align_faces.py --src <input_folder> --dst <output_folder> --size 224

Options:
    --src       : Source folder containing raw images
    --dst       : Destination folder for aligned images
    --size      : Crop size (224 for standard model, 512 for HQ model)
    --threshold : Face detection confidence threshold (default: 0.6)
    --single    : Only keep the largest face per image (default: True)
    --recursive : Process subfolders recursively (default: False)

Output format:
    - Images are cropped and aligned to 224x224 or 512x512
    - Face is centered and aligned using 5-point landmarks
    - Suitable for VGGFace2-style training data structure
"""

import os
import sys
import cv2
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

# Add SimSwap project root to path
SIMSWAP_ROOT = r'C:\toir\SimSwap'
sys.path.insert(0, SIMSWAP_ROOT)

from insightface_func.face_detect_crop_single import Face_detect_crop as Face_detect_single
from insightface_func.face_detect_crop_multi import Face_detect_crop as Face_detect_multi

# Models path
INSIGHTFACE_MODELS_ROOT = os.path.join(SIMSWAP_ROOT, 'insightface_func', 'models')


class FaceAligner:
    def __init__(self, crop_size=224, det_thresh=0.6, ctx_id=0):
        """
        Initialize face aligner
        
        Args:
            crop_size: Output size (224 or 512)
            det_thresh: Detection confidence threshold
            ctx_id: GPU id (0 for first GPU, -1 for CPU)
        """
        self.crop_size = crop_size
        self.det_thresh = det_thresh
        
        # Set mode based on crop size
        if crop_size == 512:
            self.mode = 'ffhq'
        else:
            self.mode = 'None'
        
        # Initialize face detector (use multi to get all faces)
        print("Loading face detection models...")
        self.app = Face_detect_multi(name='antelope', root=INSIGHTFACE_MODELS_ROOT)
        self.app.prepare(ctx_id=ctx_id, det_thresh=det_thresh, det_size=(640, 640), mode=self.mode)
        print("Face detection models loaded!")
    
    def process_image(self, img_path, keep_single=True):
        """
        Process a single image and return aligned faces
        
        Args:
            img_path: Path to input image
            keep_single: If True, only return the largest face
            
        Returns:
            List of (aligned_face, confidence) tuples, or None if no face found
        """
        img = cv2.imread(img_path)
        if img is None:
            return None
        
        # Detect and align faces
        results = self.app.get(img, self.crop_size)
        
        if results is None:
            return None
        
        aligned_faces, matrices = results
        
        if keep_single and len(aligned_faces) > 1:
            # Keep only the largest face (by area in original detection)
            # Since all are cropped to same size, we pick the first one 
            # which is typically the most prominent/largest
            aligned_faces = [aligned_faces[0]]
        
        return aligned_faces
    
    def process_folder(self, src_dir, dst_dir, keep_single=True, recursive=False):
        """
        Process all images in a folder
        
        Args:
            src_dir: Source directory
            dst_dir: Destination directory
            keep_single: Keep only largest face per image
            recursive: Process subfolders
        """
        valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff', '.gif')
        
        os.makedirs(dst_dir, exist_ok=True)
        
        # Collect all image files
        image_files = []
        
        if recursive:
            for root, dirs, files in os.walk(src_dir):
                for f in files:
                    if f.lower().endswith(valid_exts):
                        rel_path = os.path.relpath(root, src_dir)
                        image_files.append((os.path.join(root, f), rel_path))
        else:
            for f in os.listdir(src_dir):
                if f.lower().endswith(valid_exts):
                    image_files.append((os.path.join(src_dir, f), '.'))
        
        print(f"Found {len(image_files)} images to process")
        
        stats = {
            'processed': 0,
            'success': 0,
            'no_face': 0,
            'error': 0,
            'total_faces': 0
        }
        
        for img_path, rel_dir in tqdm(image_files, desc="Aligning faces"):
            filename = os.path.basename(img_path)
            name_without_ext = os.path.splitext(filename)[0]
            
            # Create output subdirectory if needed
            if rel_dir != '.':
                out_subdir = os.path.join(dst_dir, rel_dir)
                os.makedirs(out_subdir, exist_ok=True)
            else:
                out_subdir = dst_dir
            
            try:
                aligned_faces = self.process_image(img_path, keep_single)
                stats['processed'] += 1
                
                if aligned_faces is None or len(aligned_faces) == 0:
                    stats['no_face'] += 1
                    continue
                
                # Save aligned faces
                for i, face in enumerate(aligned_faces):
                    if len(aligned_faces) > 1:
                        out_name = f"{name_without_ext}_{i}.jpg"
                    else:
                        out_name = f"{name_without_ext}.jpg"
                    
                    out_path = os.path.join(out_subdir, out_name)
                    cv2.imwrite(out_path, face, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    stats['total_faces'] += 1
                
                stats['success'] += 1
                
            except Exception as e:
                stats['error'] += 1
                print(f"\nError processing {filename}: {e}")
        
        return stats


def create_identity_folders(src_dir, dst_dir, aligner, min_images=5):
    """
    Create VGGFace2-style folder structure with one folder per identity
    
    This function expects images to be named with a pattern like:
    - person001_001.jpg, person001_002.jpg, ...
    - Or organized in subfolders already
    
    Args:
        src_dir: Source directory
        dst_dir: Destination directory  
        aligner: FaceAligner instance
        min_images: Minimum images per identity to include
    """
    print("Creating identity-based folder structure...")
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    # Group images by identity (prefix before last underscore or folder name)
    identities = {}
    
    for root, dirs, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        
        for f in files:
            if not f.lower().endswith(valid_exts):
                continue
            
            img_path = os.path.join(root, f)
            
            # Determine identity
            if rel_path != '.':
                # If in subfolder, use folder name as identity
                identity = rel_path.replace(os.sep, '_')
            else:
                # Extract identity from filename (everything before last underscore)
                name = os.path.splitext(f)[0]
                parts = name.rsplit('_', 1)
                identity = parts[0] if len(parts) > 1 else name
            
            if identity not in identities:
                identities[identity] = []
            identities[identity].append(img_path)
    
    print(f"Found {len(identities)} identities")
    
    # Filter by minimum images
    valid_identities = {k: v for k, v in identities.items() if len(v) >= min_images}
    print(f"Identities with >= {min_images} images: {len(valid_identities)}")
    
    total_stats = {
        'identities': 0,
        'total_faces': 0
    }
    
    for identity, img_paths in tqdm(valid_identities.items(), desc="Processing identities"):
        identity_dir = os.path.join(dst_dir, identity)
        os.makedirs(identity_dir, exist_ok=True)
        
        face_count = 0
        for img_path in img_paths:
            try:
                aligned_faces = aligner.process_image(img_path, keep_single=True)
                if aligned_faces:
                    for face in aligned_faces:
                        out_path = os.path.join(identity_dir, f"{face_count:04d}.jpg")
                        cv2.imwrite(out_path, face, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        face_count += 1
            except Exception as e:
                continue
        
        if face_count >= min_images:
            total_stats['identities'] += 1
            total_stats['total_faces'] += face_count
        else:
            # Remove identity folder if not enough faces
            import shutil
            shutil.rmtree(identity_dir, ignore_errors=True)
    
    return total_stats


def main():
    parser = argparse.ArgumentParser(
        description='Align faces for SimSwap fine-tuning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage - align all faces from input folder
  python align_faces.py --src ./raw_images --dst ./aligned_images

  # Use 512x512 for high-quality model
  python align_faces.py --src ./raw_images --dst ./aligned_images --size 512

  # Keep all faces in images (not just the largest)
  python align_faces.py --src ./raw_images --dst ./aligned_images --all-faces

  # Process subfolders recursively
  python align_faces.py --src ./raw_images --dst ./aligned_images --recursive

  # Create VGGFace2-style identity folders
  python align_faces.py --src ./raw_images --dst ./aligned_images --identity-mode
        """
    )
    
    parser.add_argument('--src', type=str, required=True,
                        help='Source directory containing raw images')
    parser.add_argument('--dst', type=str, required=True,
                        help='Destination directory for aligned images')
    parser.add_argument('--size', type=int, default=224, choices=[224, 512],
                        help='Output crop size (224 or 512)')
    parser.add_argument('--threshold', type=float, default=0.6,
                        help='Face detection confidence threshold')
    parser.add_argument('--all-faces', action='store_true',
                        help='Keep all faces in each image (default: only largest)')
    parser.add_argument('--recursive', action='store_true',
                        help='Process subfolders recursively')
    parser.add_argument('--identity-mode', action='store_true',
                        help='Create VGGFace2-style identity folders')
    parser.add_argument('--min-images', type=int, default=5,
                        help='Minimum images per identity (for identity-mode)')
    parser.add_argument('--cpu', action='store_true',
                        help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    # Validate paths
    if not os.path.exists(args.src):
        print(f"Error: Source directory does not exist: {args.src}")
        sys.exit(1)
    
    # Initialize aligner
    ctx_id = -1 if args.cpu else 0
    aligner = FaceAligner(
        crop_size=args.size,
        det_thresh=args.threshold,
        ctx_id=ctx_id
    )
    
    print(f"\nSettings:")
    print(f"  Source: {args.src}")
    print(f"  Destination: {args.dst}")
    print(f"  Crop size: {args.size}x{args.size}")
    print(f"  Detection threshold: {args.threshold}")
    print(f"  Keep single face: {not args.all_faces}")
    print(f"  Recursive: {args.recursive}")
    print(f"  Device: {'CPU' if args.cpu else 'GPU'}")
    print()
    
    if args.identity_mode:
        stats = create_identity_folders(
            args.src, args.dst, aligner, 
            min_images=args.min_images
        )
        print(f"\nResults:")
        print(f"  Identities created: {stats['identities']}")
        print(f"  Total faces saved: {stats['total_faces']}")
    else:
        stats = aligner.process_folder(
            args.src, args.dst,
            keep_single=not args.all_faces,
            recursive=args.recursive
        )
        print(f"\nResults:")
        print(f"  Images processed: {stats['processed']}")
        print(f"  Successful: {stats['success']}")
        print(f"  No face detected: {stats['no_face']}")
        print(f"  Errors: {stats['error']}")
        print(f"  Total faces saved: {stats['total_faces']}")
    
    print(f"\nAligned faces saved to: {args.dst}")
    print("Done!")


if __name__ == '__main__':
    main()
