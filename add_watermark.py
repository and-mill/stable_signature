import torch
import fire
from pathlib import Path

from watermark_utils import load_model, decode_message_pil, eval_message

from PIL import Image
import torch
import torchvision.transforms as transforms
import numpy as np

from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def main(device=0, path="/USERSPACE/lukovdg1/coco2017/train2017/", outputdir="coco_wm_1000", outputdir_original=None, maximg=1000, min_res_filter=512):
    device = torch.device("cuda", device)
    
    print("start")
    pipe = load_model(device, use_watermarked=True)
    msg_extractor = torch.jit.load("models/dec_48b_whit.torchscript.pt").to(device)
    print("loaded")
    
    numimg = 0

    for spath in Path(path).glob("*"):
        img = Image.open(spath).convert("RGB")
        if min(img.size) < min_res_filter:
            continue
        
        img = transforms.CenterCrop(min(img.size))(img)
        img = transforms.Resize(768)(img)
        
        print(f"Doing {spath}")
        orig_img = img
        img = (transforms.ToTensor()(img) - 0.5)/0.5
        img = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            z = pipe.vae.encode(img)
            reimg = pipe.vae.decode(z.latent_dist.mode())[0]
            reimg = reimg[0] * 0.5 + 0.5
        
        # print("decoding message")
        reimg = transforms.ToPILImage()(reimg.clamp(0, 1))
        msg = decode_message_pil(reimg, msg_extractor=msg_extractor, device=device)
        print(f"decoded message: {msg}")
        eval_message(msg)
        
        psnr = peak_signal_noise_ratio(np.asarray(orig_img), np.asarray(reimg))
        ssim = structural_similarity(np.asarray(orig_img), np.asarray(reimg), channel_axis=2)
        print(f"PSNR: {psnr}, SSIM: {ssim}")
        
        if outputdir is not None:
            outputpath = Path(outputdir) / (spath.stem + ".wm.png")
            if not outputpath.parent.exists():
                Path.mkdir(outputpath.parent, parents=True, exist_ok=False)
            reimg.save(outputpath)
            
        if outputdir_original is not None:
            outputpath = Path(outputdir_original) / (spath.stem + ".original.png")
            if not outputpath.parent.exists():
                Path.mkdir(outputpath.parent, parents=True, exist_ok=False)
            reimg.save(outputpath)
            
        numimg += 1
        if numimg == maximg:
            break


if __name__ == "__main__":
    fire.Fire(main)