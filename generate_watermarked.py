from pytorch_lightning import seed_everything
import torch
import fire

from watermark_utils import load_model, decode_message_pil, eval_message


def main(device=0, prompt="the cat drinks water.", savepath="cat.no_wm.png", use_watermark=True, seed=None):
    device = torch.device("cuda", device)
    
    if seed is not None:
        seed_everything(seed)
    
    print("start")
    pipe = load_model(device, use_watermarked=use_watermark)
    print("loaded")

    print("generating")
    img = pipe(prompt).images[0]
    if savepath is not None:
        img.save(savepath)
    print("generated")
    
    print("decoding message")
    msg = decode_message_pil(img)
    print(f"decoded message: {msg}")
    eval_message(msg)


if __name__ == "__main__":
    fire.Fire(main)