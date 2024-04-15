import torch
import fire
import re

from omegaconf import OmegaConf 
from diffusers import StableDiffusionPipeline 
from utils_model import load_model_from_config 

from PIL import Image
import torch
import torchvision.transforms as transforms


KEY = '111010110101000001010111010011010100010000100111' # model key


def msg2str(msg):
    return "".join([('1' if el else '0') for el in msg])


def str2msg(str):
    return [True if el=='1' else False for el in str]


def decode_message_path(path):
    transform_imnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    img = Image.open("path/to/generated/img.png")
    img = transform_imnet(img).unsqueeze(0).to("cuda")
    return decode_message(img)
    
    
def decode_message_pil(pilimg):
    transform_imnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    img = transform_imnet(pilimg).unsqueeze(0).to("cuda")
    return decode_message(img)


def decode_message(img):
    msg_extractor = torch.jit.load("models/dec_48b_whit.torchscript.pt").to("cuda")
    msg = msg_extractor(img) # b c h w -> b k
    bool_msg = (msg>0).squeeze().cpu().numpy().tolist()
    print("Extracted message: ", msg2str(bool_msg))
    return bool_msg


def eval_message(msg):
    bool_key = str2msg(KEY)

    # compute difference between model key and message extracted from image
    diff = [msg[i] != bool_key[i] for i in range(len(msg))]
    bit_acc = 1 - sum(diff)/len(diff)
    print("Bit accuracy: ", bit_acc)

    # compute p-value
    from scipy.stats import binomtest
    pval = binomtest(len(diff)-sum(diff), len(diff), 0.5, alternative='greater')
    print("p-value of statistical test: ", pval)
    
    return bit_acc, pval


def load_model(device, use_watermarked=True):
    # load original diffusers pipe
    model = "stabilityai/stable-diffusion-2"
    pipe = StableDiffusionPipeline.from_pretrained(model).to(device)
    
    if use_watermarked:
        # load stable-diffusion codebase decoder
        ldm_config = "sd/stable-diffusion-2-1-base/v2-inference.yaml"
        ldm_ckpt = "sd/stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt"

        print(f'>>> Building LDM model with config {ldm_config} and weights from {ldm_ckpt}...')
        config = OmegaConf.load(f"{ldm_config}")
        ldm_ae = load_model_from_config(config, ldm_ckpt)
        ldm_aef = ldm_ae.first_stage_model
        ldm_aef.eval()
        
        state_dict = torch.load("models/sd2_decoder.pth")
        unexpected_keys = ldm_aef.load_state_dict(state_dict, strict=False)
        print(unexpected_keys)
        print("you should check that the decoder keys are correctly matched")
        
        # replace decode of vae in diffusers pipe with custom LDM decoder 
        pipe.vae.decode = (lambda x,  *args, **kwargs: ldm_aef.decode(x).unsqueeze(0))
        
    return pipe


def main(device=0, path="cat.no_wm.png", savepath="cat.wma.png"):
    device = torch.device("cuda", device)
    
    print("start")
    pipe = load_model(device, use_watermarked=True)
    print("loaded")

    transform_imnet = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(path)
    img = transform_imnet(img).unsqueeze(0).to("cuda")
    
    z = pipe.vae.encode(img)
    reimg = pipe.vae.decode(z.latent_dist.mode())[0]
    
    print("decoding message")
    msg = decode_message(reimg)
    print(f"decoded message: {msg}")
    eval_message(msg)


if __name__ == "__main__":
    fire.Fire(main)