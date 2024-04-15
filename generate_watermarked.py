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


def remap_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        assert(k.startswith("decoder."))
        k = k[len("decoder."):]
        
        if k.startswith("conv_in.") or k.startswith("conv_out."):
            pass
        
        if k.startswith("norm_out."):
            ks = k.split("norm_out.")[1]
            k = "conv_norm_out." + ks
            
        if k.startswith("mid.block_1"):
            ks = k.split("mid.block_1")[1]
            k = "mid_block.resnets.0" + ks
            
        if k.startswith("mid.block_2"):
            ks = k.split("mid.block_2")[1]
            k = "mid_block.resnets.1" + ks
            
        if k.startswith("mid.attn_1"):
            ks = k.split("mid.attn_1")[1]
            ks = ks.replace("q.", "to_q.")
            ks = ks.replace("k.", "to_k.")
            ks = ks.replace("v.", "to_v.")
            ks = ks.replace("proj_out.", "to_out.")
            ks = ks.replace("norm.", "group_norm.")
            k = "mid_block.attentions.0" + ks
            
        if k.startswith("up."):
            ks = k.split("up.")
            k = re.sub("^up\.(\d)\.block.(\d).", "up_blocks.\g<1>.resnets.\g<2>.", k)
            ks = None
            k = k.replace("nin_shortcut", "conv_shortcut")
            
            # TODO
            
        
        new_state_dict[k] = v    
    return new_state_dict


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


def main(device=0, prompt="the cat drinks water.", savepath="cat.no_wm.png", use_watermark=False):
    device = torch.device("cuda", device)
    
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