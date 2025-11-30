"""
Simple test script for SimSwap
Swap face from pic_a to pic_b
"""

import cv2
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
from torchvision import transforms
from models.models import create_model
from options.test_options import TestOptions
from insightface_func.face_detect_crop_single import Face_detect_crop
from util.reverse2original import reverse2wholeimage
from util.add_watermark import watermark_image
from util.norm import SpecificNorm
from parsing_model.model import BiSeNet
import os

transformer_Arcface = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def _totensor(array):
    tensor = torch.from_numpy(array)
    img = tensor.transpose(0, 1).transpose(0, 2).contiguous()
    return img.float().div(255)

if __name__ == '__main__':
    opt = TestOptions().parse()
    
    crop_size = opt.crop_size
    
    # Setup model
    torch.nn.Module.dump_patches = True
    if crop_size == 512:
        opt.which_epoch = 550000
        opt.name = '512'
        mode = 'ffhq'
    else:
        mode = 'None'
    
    print(f"Loading model: {opt.name}")
    model = create_model(opt)
    model.eval()
    
    # Setup face detection
    print("Loading face detection model...")
    spNorm = SpecificNorm()
    app = Face_detect_crop(name='antelope', root='./insightface_func/models')
    app.prepare(ctx_id=0, det_thresh=0.6, det_size=(640,640), mode=mode)
    
    # Logo (optional)
    logoclass = watermark_image('./simswaplogo/simswaplogo.png')
    
    with torch.no_grad():
        # Load source face (the face you want to use)
        print(f"Loading source face from: {opt.pic_a_path}")
        img_a_whole = cv2.imread(opt.pic_a_path)
        if img_a_whole is None:
            print(f"ERROR: Cannot read {opt.pic_a_path}")
            exit(1)
            
        img_a_align_crop, _ = app.get(img_a_whole, crop_size)
        if img_a_align_crop is None:
            print(f"ERROR: No face detected in {opt.pic_a_path}")
            exit(1)
            
        img_a_align_crop_pil = Image.fromarray(cv2.cvtColor(img_a_align_crop[0], cv2.COLOR_BGR2RGB))
        img_a = transformer_Arcface(img_a_align_crop_pil)
        img_id = img_a.view(-1, img_a.shape[0], img_a.shape[1], img_a.shape[2])
        img_id = img_id.cuda()
        
        # Extract identity features
        print("Extracting identity features...")
        img_id_downsample = F.interpolate(img_id, size=(112,112))
        latend_id = model.netArc(img_id_downsample)
        latend_id = F.normalize(latend_id, p=2, dim=1)
        
        # Load target image (the image where you want to swap face)
        print(f"Loading target image from: {opt.pic_b_path}")
        img_b_whole = cv2.imread(opt.pic_b_path)
        if img_b_whole is None:
            print(f"ERROR: Cannot read {opt.pic_b_path}")
            exit(1)
            
        img_b_align_crop_list, b_mat_list = app.get(img_b_whole, crop_size)
        if img_b_align_crop_list is None:
            print(f"ERROR: No face detected in {opt.pic_b_path}")
            exit(1)
            
        print(f"Found {len(img_b_align_crop_list)} face(s) in target image")
        
        # Perform face swap
        print("Swapping faces...")
        swap_result_list = []
        b_align_crop_tenor_list = []
        
        for b_align_crop in img_b_align_crop_list:
            b_align_crop_tenor = _totensor(cv2.cvtColor(b_align_crop, cv2.COLOR_BGR2RGB))[None,...].cuda()
            swap_result = model(None, b_align_crop_tenor, latend_id, None, True)[0]
            swap_result_list.append(swap_result)
            b_align_crop_tenor_list.append(b_align_crop_tenor)
        
        # Load face parsing model if using mask
        if opt.use_mask:
            print("Loading face parsing model...")
            n_classes = 19
            net = BiSeNet(n_classes=n_classes)
            net.cuda()
            save_pth = os.path.join('./parsing_model/checkpoint', '79999_iter.pth')
            net.load_state_dict(torch.load(save_pth))
            net.eval()
        else:
            net = None
        
        # Merge result back to original image
        print("Merging result...")
        os.makedirs(opt.output_path, exist_ok=True)
        output_file = os.path.join(opt.output_path, 'result.jpg')
        
        reverse2wholeimage(
            b_align_crop_tenor_list, 
            swap_result_list, 
            b_mat_list, 
            crop_size, 
            img_b_whole, 
            logoclass,
            output_file, 
            opt.no_simswaplogo,
            pasring_model=net,
            use_mask=opt.use_mask, 
            norm=spNorm
        )
        
        print(f"Done! Result saved to: {output_file}")
