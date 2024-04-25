import fire

import torch
#from transformers import AdamW
from torch.optim import AdamW
from diffusers import AutoencoderKL

import utils
import utils_img
import utils_model
from watermark_utils import decode_message_pil, eval_message, msg2str, str2msg, load_model

from torchvision import transforms
from torchvision.utils import save_image

import sys
from omegaconf import OmegaConf

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

import utils
import utils_img
import utils_model

sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion
from loss.loss_provider import LossProvider

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

KEY = '111010110101000001010111010011010100010000100111'
LOSS_W = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2) # b k - b k


def main(ldm_config: str = "/home/host_mueller/mueller/git/stable_signature/sd/stable-diffusion-2-1-base/v2-inference.yaml",
         ldm_ckpt: str = "/home/host_mueller/mueller/git/stable_signature/sd/stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt",
         ldm_ckpt_watermarked: str = "/home/host_mueller/mueller/git/stable_signature/models/sd2_decoder.pth",
         msg_extractor_path: str = "/home/host_mueller/mueller/git/stable_signature/models/dec_48b_whit.torchscript.pt",
         train_dir: str = "/home/host_datasets/COCO/train2017",
         #val_dir: str = "/home/host_datasets/COCO/val2017",
         img_size: int = 256,
         batch_size: int = 4,
         epochs: int = 4) -> None:
    
    # ----------------- mostly copied from finetune_ldm_decoder.py -----------------

    """
    # Loads LDM auto-encoder models
    config = OmegaConf.load(f"{ldm_config}")
    ldm_ae: LatentDiffusion = utils_model.load_model_from_config(config, ldm_ckpt)
    ldm_ae: AutoencoderKL = ldm_ae.first_stage_model
    # Load the watermarked decoder
    watermarked_decoder_state_dict = torch.load(ldm_ckpt_watermarked)#['ldm_decoder']
    _ = ldm_ae.load_state_dict(watermarked_decoder_state_dict, strict=False)

    # Load msg decoder (HiDDeN). Assume its the torchscript variant
    msg_extractor = torch.jit.load(msg_extractor_path).to(device)
    msg_extractor.eval()
    nbit = msg_extractor(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]

    # Freeze LDM and hidden decoder
    for param in [*msg_extractor.parameters(), *ldm_ae.encoder.parameters()]:
        param.requires_grad = False
    # prepare
    ldm_ae.eval()
    ldm_ae.to(device)
    """

    """
    # Test the message extraction by generating an image and checking extracted message
    pipe = load_model(device, use_watermarked=True)
    img = pipe('fish playing the piano').images[0]
    img.save("temp.png")
    msg = decode_message_pil(img)
    print(f"decoded message: {msg}")
    eval_message(msg, verbose=True)
    """

    # get model
    ldm_ae = pipe.vae

    # Loads the data
    vqgan_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    train_loader = utils.get_dataloader(train_dir, vqgan_transform, batch_size, num_imgs=batch_size * epochs, shuffle=True, num_workers=4, collate_fn=None)
    #val_loader = utils.get_dataloader(val_dir, vqgan_transform, batch_size*4, num_imgs=1000, shuffle=False, num_workers=4, collate_fn=None)
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])

    # ----------------- mostly copied from finetune_ldm_decoder.py -----------------

    # Configure the optimizer
    optimizer = AdamW(ldm_ae.parameters(), lr=2e-4)
    
    # Training loop
    for param in ldm_ae.decoder.parameters():
            param.requires_grad = True
    ldm_ae.decoder.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            
            batch=batch.to(device)
    
            # encode images
            imgs_z = ldm_ae.encode(batch) # b c h w -> b z h/f w/f
            imgs_z = imgs_z.mode()
            
            # decode latents with original and finetuned decoder
            imgs_d0 = ldm_ae.decode(imgs_z) # b z h/f w/f -> b c h w

            # compute loss
            loss = nn.MSELoss()(imgs_d0, batch)  # This is used in stable diffusion AE training

            # optim step
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # extract watermark
            keys = torch.tensor(str2msg(KEY)).to(device).repeat(batch.shape[0], 1)
            msg_extracted = msg_extractor(vqgan_to_imnet(batch)) # b c h w -> b k

            #bitacc, pval = eval_message(msg_extracted)
            #print(np.mean(bitacc))

            diff = (~torch.logical_xor(msg_extracted>0, keys>0)) # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            print(f"Bit accuracy: {torch.mean(bit_accs).item()}")
    
            total_loss += loss.item()
    
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch}, Average Loss: {avg_loss:.4f}")
    
    print("Retraining complete")


if __name__ == "__main__":
    fire.Fire(main)
