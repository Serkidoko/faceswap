import os
import cv2
import argparse
from insightface_func.face_detect_crop_multi import Face_detect_crop

def preprocess(src_dir, dst_dir, crop_size=224):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    
    # Ensure the model path is correct
    # The Face_detect_crop expects 'root' to contain the 'name' folder.
    # So if we pass root='./insightface_func/models', it looks for './insightface_func/models/antelope'
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    
    # ctx_id=0 for GPU, -1 for CPU
    # We assume GPU is available as per requirements, but let's be safe or allow config.
    # SimSwap requires GPU usually.
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640))
    
    valid_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    count = 0
    for filename in os.listdir(src_dir):
        if filename.lower().endswith(valid_exts):
            img_path = os.path.join(src_dir, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Could not read {filename}")
                continue
            
            try:
                results = app.get(img, crop_size)
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue

            if results is None:
                print(f"No face detected in {filename}")
                continue
            
            align_img_list, _ = results
            
            for i, align_img in enumerate(align_img_list):
                # If multiple faces, append index. If single, maybe just name?
                # But to be safe, always append index or check count.
                if len(align_img_list) > 1:
                    save_name = f"{os.path.splitext(filename)[0]}_{i}.jpg"
                else:
                    save_name = f"{os.path.splitext(filename)[0]}.jpg"
                
                save_path = os.path.join(dst_dir, save_name)
                cv2.imwrite(save_path, align_img)
                count += 1
                if count % 10 == 0:
                    print(f"Processed {count} images...")

    print(f"Finished. Total aligned images: {count}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, required=True, help='Source directory containing raw images')
    parser.add_argument('--dst', type=str, required=True, help='Destination directory for aligned images')
    parser.add_argument('--size', type=int, default=224, help='Crop size (224 or 512)')
    args = parser.parse_args()
    
    preprocess(args.src, args.dst, args.size)
