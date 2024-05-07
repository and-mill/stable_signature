import torch
import fire
import numpy as np


from pathlib import Path

from watermark_utils import decode_message_pil, eval_message

from PIL import Image


def main(device=1, path="images/coco_1000_iDDIM_1of100/*"):
    device = torch.device("cuda", device)
    
    msg_extractor = torch.jit.load("models/dec_48b_whit.torchscript.pt").to(device)
    
    bitaccs, pvals = [], []
    for spath in Path().glob(path):
        print(f"Doing file: {spath}")
        img = Image.open(spath)
        
        print("decoding message")
        msg = decode_message_pil(img, msg_extractor=msg_extractor, device=device)
        # print(f"decoded message: {msg}")
        bitacc, pval = eval_message(msg)
        bitaccs.append(bitacc); pvals.append(pval)
        
    print(f"Average bit acc.: {np.mean(bitaccs)}")


if __name__ == "__main__":
    fire.Fire(main)