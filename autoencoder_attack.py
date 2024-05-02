from pathlib import Path
from diffusers import StableDiffusionPipeline , KandinskyPipeline
import fire
import torch
import tqdm 
from PIL import Image
from torchvision import transforms


def run(ae="kd2.1", 
        inpath="/USERSPACE/lukovdg1/stable_signature/images/coco_wm_1000/", 
        outpath="/USERSPACE/lukovdg1/stable_signature/images/coco_1000_ae_", 
        device=1):
    if ae.startswith("sd"):
        sd_model_ckpt = {"sd1.5": "runwayml/stable-diffusion-v1-5",
                        "sd2.1": "stabilityai/stable-diffusion-2"}[ae]
        pipe = StableDiffusionPipeline.from_pretrained(sd_model_ckpt, safety_checker=None).to(torch.device("cuda", device))
        print("loaded model")
        
    elif ae.startswith("kd"):
        kd_model_ckpt = {"kd2.1": "kandinsky-community/kandinsky-2-1"}[ae]
        pipe = KandinskyPipeline.from_pretrained(kd_model_ckpt).to(torch.device("cuda", device))
        
    outpath = outpath + ae
    Path(outpath).mkdir(parents=True, exist_ok=True)
        
    for i, impath in enumerate(tqdm.tqdm(Path(inpath).glob("*"))):
        print(f"Doing image {i}")
        img = Image.open(impath).convert("RGB")
        img = (transforms.ToTensor()(img) - 0.5)/0.5
        img = img.unsqueeze(0).to(device)
        # print("image size:", rawimage.size)
        
        with torch.no_grad():
            if ae.startswith("sd"):
                z = pipe.vae.encode(img)
                reimg = pipe.vae.decode(z.latent_dist.mode())[0]
            elif ae.startswith("kd"):
                z = pipe.movq.encode(img)
                reimg = pipe.movq.decode(z.latents).sample
                    
            reimg = reimg[0] * 0.5 + 0.5
            
        reimg = transforms.ToPILImage()(reimg.clamp(0, 1))
        
        _outpath = Path(outpath) / Path(impath).name
        reimg.save(_outpath)
        


if __name__ == "__main__":  
    fire.Fire(run)