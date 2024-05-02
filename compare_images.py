import torch
import fire
from pathlib import Path

from watermark_utils import load_model, decode_message_pil, eval_message

from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def main(device=2, 
         originalpath="images/coco_wm_1000/", 
        #  originalpath="images/coco_original_1000/", 
        #  purifiedpath="images/coco_1000_iDDIM_1of100/",
        #  purifiedpath="images/coco_1000_ae_sd1.5/",
         purifiedpath="images/coco_1000_ae_kd2.1/",
        #  purifiedpath="images/coco_wm_1000/"
         ):
    device = torch.device("cuda", device)
    
    print("start")
    msg_extractor = torch.jit.load("models/dec_48b_whit.torchscript.pt").to(device)
    print("loaded")
    
    avgacc = []
    avgpsnr, avgssim = [], []
    for i, spath in enumerate(Path(purifiedpath).glob("*")):
        img = Image.open(spath).convert("RGB")
        
        ref_img_path = Path(originalpath) / spath.name
        assert ref_img_path.exists()
        ref_img = Image.open(ref_img_path).convert("RGB")
        
        msg = decode_message_pil(img, msg_extractor=msg_extractor, device=device, verbose=False)
        bitacc, _ = eval_message(msg, verbose=False)
        
        psnr = peak_signal_noise_ratio(np.asarray(ref_img), np.asarray(img))
        ssim = structural_similarity(np.asarray(ref_img), np.asarray(img), channel_axis=2)
        print(f"Doing image nr. {i}: {spath}: Bitacc: {bitacc:.3f}, PSNR: {psnr:.3f}, SSIM: {ssim:.3f}")
        
        avgacc.append(bitacc)
        avgpsnr.append(psnr)
        avgssim.append(ssim)
        
    print(f"Average bit acc: {np.mean(avgacc):.3f}, average PSNR: {np.mean(avgpsnr):.3f}, average SSIM: {np.mean(avgssim):.3f}")
        
            


if __name__ == "__main__":
    fire.Fire(main)