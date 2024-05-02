from pathlib import Path
from pytorch_lightning import seed_everything
import torch
import fire

from watermark_utils import load_model, decode_message_pil, eval_message


# TODO: generated 1000 images and repeat AE attacks

def main(device=3, prompts="the cat drinks water.", savedir="", savenames="cat.no_wm", use_watermark=True, seed=None):
    prompts = prompts.split(",")
    savenames = savenames.split(",")
    print(device)
    device = torch.device("cuda", device)
    
    if seed is not None:
        seed_everything(seed)
    
    print("start")
    pipe = load_model(device, use_watermarked=use_watermark)
    print("loaded")
    
    for prompt, savename in zip(prompts, savenames):
        savepath = Path(savedir) / (savename + f".{seed}.png")

        print(f"generating for prompt {prompt}")
        img = pipe(prompt).images[0]
        if savepath is not None:
            print(f"saving in: '{savepath}'")
            img.save(savepath)
        print("generated")
        
        print("decoding message")
        msg = decode_message_pil(img)
        eval_message(msg)


if __name__ == "__main__":
    fire.Fire(main)