import os
import requests
import zipfile
import tarfile
import gdown
import shutil

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    if os.path.exists(dest_path):
        print(f"{dest_path} already exists. Skipping.")
        return

    try:
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)

def unzip_file(zip_path, extract_to):
    print(f"Unzipping {zip_path} to {extract_to}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print("Unzip complete.")
    except Exception as e:
        print(f"Failed to unzip {zip_path}: {e}")

def untar_file(tar_path, extract_to):
    print(f"Extracting {tar_path} to {extract_to}...")
    try:
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_to)
        print("Extraction complete.")
    except Exception as e:
        print(f"Failed to extract {tar_path}: {e}")

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. ArcFace Checkpoint
    arcface_dir = os.path.join(base_dir, 'arcface_model')
    os.makedirs(arcface_dir, exist_ok=True)
    arcface_url = "https://github.com/neuralchen/SimSwap/releases/download/1.0/arcface_checkpoint.tar"
    arcface_path = os.path.join(arcface_dir, 'arcface_checkpoint.tar')
    download_file(arcface_url, arcface_path)
    # Note: The tar file is usually used as is or extracted? 
    # The README says "Copy the arcface_checkpoint.tar into ./arcface_model". It doesn't say extract.
    # But usually tar files are archives. Let's check if it needs extraction.
    # The code usually loads the file directly if it's a checkpoint, but .tar usually implies archive.
    # However, torch.load can load tar if it's just a renamed pth, but usually it's an archive.
    # Let's leave it as is for now as per README instructions.

    # 2. SimSwap Checkpoints
    checkpoints_url = "https://github.com/neuralchen/SimSwap/releases/download/1.0/checkpoints.zip"
    checkpoints_zip = os.path.join(base_dir, 'checkpoints.zip')
    download_file(checkpoints_url, checkpoints_zip)
    unzip_file(checkpoints_zip, base_dir) # Extracts to ./checkpoints
    
    # 3. InsightFace (Antelope replacement: buffalo_l)
    # Original antelope link is dead. Using buffalo_l from insightface issue #1896
    # https://drive.google.com/file/d/1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB/view?usp=sharing
    buffalo_l_id = "1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB"
    insightface_models_dir = os.path.join(base_dir, 'insightface_func', 'models')
    antelope_dir = os.path.join(insightface_models_dir, 'antelope')
    os.makedirs(antelope_dir, exist_ok=True)
    
    # Check if we already have models
    if not os.listdir(antelope_dir):
        print("Downloading buffalo_l (as antelope replacement)...")
        buffalo_zip = os.path.join(base_dir, 'buffalo_l.zip')
        try:
            gdown.download(id=buffalo_l_id, output=buffalo_zip, quiet=False)
            unzip_file(buffalo_zip, antelope_dir)
            # buffalo_l zip might contain a folder 'buffalo_l', we need to move contents up if so
            # Let's check structure after unzip
            if os.path.exists(os.path.join(antelope_dir, 'buffalo_l')):
                print("Adjusting folder structure...")
                src = os.path.join(antelope_dir, 'buffalo_l')
                for item in os.listdir(src):
                    shutil.move(os.path.join(src, item), antelope_dir)
                os.rmdir(src)
        except Exception as e:
            print(f"Failed to download buffalo_l: {e}")
    else:
        print("insightface_func/models/antelope is not empty. Skipping download.")
    
    # 4. Face Parsing
    parsing_dir = os.path.join(base_dir, 'parsing_model', 'checkpoint')
    os.makedirs(parsing_dir, exist_ok=True)
    parsing_id = "154JgKpzCPW82qINcVieuPH3fZ2e0P812"
    parsing_path = os.path.join(parsing_dir, '79999_iter.pth')
    
    if not os.path.exists(parsing_path):
        print(f"Downloading Face Parsing model to {parsing_path}...")
        try:
            gdown.download(id=parsing_id, output=parsing_path, quiet=False)
        except Exception as e:
            print(f"Failed to download Face Parsing model: {e}")
    else:
        print(f"{parsing_path} already exists.")

if __name__ == "__main__":
    main()
