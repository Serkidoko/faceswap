import insightface
import os
import shutil

def download_and_setup_models():
    # This will trigger download of buffalo_l to ~/.insightface/models/buffalo_l
    print("Attempting to download buffalo_l model via insightface...")
    app = insightface.app.FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    # We don't need to prepare, just init triggers download if not present
    
    # Find where it was downloaded
    user_home = os.path.expanduser('~')
    insightface_home = os.path.join(user_home, '.insightface', 'models')
    buffalo_l_path = os.path.join(insightface_home, 'buffalo_l')
    
    if os.path.exists(buffalo_l_path):
        print(f"Model found at {buffalo_l_path}")
        
        # Destination in SimSwap
        simswap_models_dir = os.path.abspath('./insightface_func/models')
        dest_path = os.path.join(simswap_models_dir, 'antelope')
        
        if not os.path.exists(dest_path):
            print(f"Copying to {dest_path}...")
            shutil.copytree(buffalo_l_path, dest_path)
            print("Copy complete.")
        else:
            print(f"Destination {dest_path} already exists.")
            
            # Check if empty
            if not os.listdir(dest_path):
                print("Destination is empty. Copying files...")
                for item in os.listdir(buffalo_l_path):
                    s = os.path.join(buffalo_l_path, item)
                    d = os.path.join(dest_path, item)
                    if os.path.isdir(s):
                        shutil.copytree(s, d)
                    else:
                        shutil.copy2(s, d)
                print("Copy complete.")
    else:
        print("Failed to download buffalo_l or could not find it.")

if __name__ == "__main__":
    download_and_setup_models()
