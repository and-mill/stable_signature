# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import os
import sys
from copy import deepcopy
from omegaconf import OmegaConf
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

import utils
import utils_img
import utils_model
import functools

sys.path.append('src')
from ldm.models.autoencoder import AutoencoderKL
from ldm.models.diffusion.ddpm import LatentDiffusion
from loss.loss_provider import LossProvider

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_parser():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        group.add_argument(*args, **kwargs)

    group = parser.add_argument_group('Data parameters')
    aa("--train_dir", default="../coco2017/train2017/", type=str, help="Path to the training data directory", required=False)
    aa("--val_dir", default="../coco2017/val2017/", type=str, help="Path to the validation data directory", required=False)

    group = parser.add_argument_group('Model parameters')
    # aa("--ldm_config", type=str, default="sd/stable-diffusion-v-1-4-original/v1-inference.yaml", help="Path to the configuration file for the LDM model") 
    # aa("--ldm_ckpt", type=str, default="sd/stable-diffusion-v-1-4-original/sd-v1-4-full-ema.ckpt", help="Path to the checkpoint file for the LDM model") 
    aa("--ldm_config", type=str, default="sd/stable-diffusion-2-1-base/v2-inference.yaml", help="Path to the configuration file for the LDM model") 
    aa("--ldm_ckpt", type=str, default="sd/stable-diffusion-2-1-base/v2-1_512-ema-pruned.ckpt", help="Path to the checkpoint file for the LDM model") 
    aa("--ckpt", type=str, default="models/sd2_decoder.pth", help="Path to the checkpoint file for the pretrained LDM decoder model") 
    aa("--key", type=str, default='111010110101000001010111010011010100010000100111', help="Key of the latent decoder loaded") 
    aa("--msg_decoder_path", type=str, default= "models/dec_48b_whit.torchscript.pt", help="Path to the hidden decoder for the watermarking model")
    aa("--num_bits", type=int, default=48, help="Number of bits in the watermark")
    aa("--redundancy", type=int, default=1, help="Number of times the watermark is repeated to increase robustness")
    aa("--decoder_depth", type=int, default=8, help="Depth of the decoder in the watermarking model")
    aa("--decoder_channels", type=int, default=64, help="Number of channels in the decoder of the watermarking model")

    group = parser.add_argument_group('Training parameters')
    aa("--batch_size", type=int, default=4, help="Batch size for training")
    aa("--img_size", type=int, default=256, help="Resize images to this size")
    aa("--loss_i", type=str, default="watson-vgg", help="Type of loss for the image loss. Can be watson-vgg, mse, watson-dft, etc.")
    aa("--loss_w", type=str, default="bce", help="Type of loss for the watermark loss. Can be mse or bce")
    aa("--lambda_i", type=float, default=1.0, help="Weight of the image loss in the final loss for the decoder")
    aa("--lambda_d", type=float, default=0.01, help="Weight of the adverarial loss in the final loss for the decoder")
    aa("--lambda_w", type=float, default=0, help="Weight of the watermark loss in the total loss")
    aa("--optimizer", type=str, default="AdamW,lr=1e-4", help="Optimizer and learning rate for training")
    aa("--steps", type=int, default=10000, help="Number of steps to train the model for")
    aa("--warmup_steps", type=int, default=200, help="Number of warmup steps for the optimizer")

    group = parser.add_argument_group('Logging and saving freq. parameters')
    aa("--log_freq", type=int, default=50, help="Logging frequency (in steps)")
    aa("--save_img_freq", type=int, default=200, help="Frequency of saving generated images (in steps)")

    group = parser.add_argument_group('Experiments parameters')
    # aa("--num_keys", type=int, default=1, help="Number of fine-tuned checkpoints to generate")
    aa("--output_dir", type=str, default="purified_decoder_10000steps_adv/", help="Output directory for logs and images (Default: /output)")
    aa("--seed", type=int, default=0)
    aa("--debug", type=utils.bool_inst, default=False, help="Debug mode")

    return parser


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class ActNorm(nn.Module):
    def __init__(self, num_features, logdet=False, affine=True,
                 allow_reverse_init=False):
        assert affine
        super().__init__()
        self.logdet = logdet
        self.loc = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.allow_reverse_init = allow_reverse_init

        self.register_buffer('initialized', torch.tensor(0, dtype=torch.uint8))

    def initialize(self, input):
        with torch.no_grad():
            flatten = input.permute(1, 0, 2, 3).contiguous().view(input.shape[1], -1)
            mean = (
                flatten.mean(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )
            std = (
                flatten.std(1)
                .unsqueeze(1)
                .unsqueeze(2)
                .unsqueeze(3)
                .permute(1, 0, 2, 3)
            )

            self.loc.data.copy_(-mean)
            self.scale.data.copy_(1 / (std + 1e-6))

    def forward(self, input, reverse=False):
        if reverse:
            return self.reverse(input)
        if len(input.shape) == 2:
            input = input[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        _, _, height, width = input.shape

        if self.training and self.initialized.item() == 0:
            self.initialize(input)
            self.initialized.fill_(1)

        h = self.scale * (input + self.loc)

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)

        if self.logdet:
            log_abs = torch.log(torch.abs(self.scale))
            logdet = height*width*torch.sum(log_abs)
            logdet = logdet * torch.ones(input.shape[0]).to(input)
            return h, logdet

        return h

    def reverse(self, output):
        if self.training and self.initialized.item() == 0:
            if not self.allow_reverse_init:
                raise RuntimeError(
                    "Initializing ActNorm in reverse direction is "
                    "disabled by default. Use allow_reverse_init=True to enable."
                )
            else:
                self.initialize(output)
                self.initialized.fill_(1)

        if len(output.shape) == 2:
            output = output[:,:,None,None]
            squeeze = True
        else:
            squeeze = False

        h = output / self.scale - self.loc

        if squeeze:
            h = h.squeeze(-1).squeeze(-1)
        return h


    
class ResnetDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        print("creating model")
        # self.model = HiddenDecoder(8, 1, 3)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.fc = torch.nn.Linear(512, 1)
        # self.model.fc = torch.nn.Identity()
        print("model created")
        
    def forward(self, x):
        ret = self.model(x)
        return ret


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator as in Pix2Pix
        --> see https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
    """
    def __init__(self, input_nc=3, ndf=64, n_layers=3, use_actnorm=False):
        """Construct a PatchGAN discriminator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if not use_actnorm:
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = ActNorm
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
            sequence += [
                nn.Conv2d(ndf * nf_mult, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.main = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        ret = self.main(input)
        return ret



def main(params):

    # Set seeds for reproductibility 
    torch.manual_seed(params.seed)
    torch.cuda.manual_seed_all(params.seed)
    np.random.seed(params.seed)
    
    # Print the arguments
    print("__git__:{}".format(utils.get_sha()))
    print("__log__:{}".format(json.dumps(vars(params))))

    # Create the directories
    if not os.path.exists(params.output_dir):
        os.makedirs(params.output_dir)
    imgs_dir = os.path.join(params.output_dir, 'imgs')
    params.imgs_dir = imgs_dir
    if not os.path.exists(imgs_dir):
        os.makedirs(imgs_dir, exist_ok=True)

    # Loads LDM auto-encoder models
    print(f'>>> Building LDM model with config {params.ldm_config} and weights from {params.ldm_ckpt}...')
    config = OmegaConf.load(f"{params.ldm_config}")
    ldm_ae: LatentDiffusion = utils_model.load_model_from_config(config, params.ldm_ckpt)
    ldm_ae: AutoencoderKL = ldm_ae.first_stage_model
    ldm_ae.load_state_dict(torch.load(params.ckpt), strict=False)
    ldm_ae.eval()
    ldm_ae.to(device)
    
    nbit = params.num_bits
    
    # """
    # Message extractor 
    # Loads hidden decoder
    print(f'>>> Building hidden decoder with weights from {params.msg_decoder_path}...')
    if 'torchscript' in params.msg_decoder_path: 
        msg_decoder = torch.jit.load(params.msg_decoder_path).to(device)
        # already whitened
        
    else:
        msg_decoder = utils_model.get_hidden_decoder(num_bits=params.num_bits, redundancy=params.redundancy, num_blocks=params.decoder_depth, channels=params.decoder_channels).to(device)
        ckpt = utils_model.get_hidden_decoder_ckpt(params.msg_decoder_path)
        print(msg_decoder.load_state_dict(ckpt, strict=False))
        msg_decoder.eval()

        # whitening
        print(f'>>> Whitening...')
        with torch.no_grad():
            # features from the dataset
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            loader = utils.get_dataloader(params.train_dir, transform, batch_size=16, collate_fn=None)
            ys = []
            for i, x in enumerate(loader):
                x = x.to(device)
                y = msg_decoder(x)
                ys.append(y.to('cpu'))
            ys = torch.cat(ys, dim=0)
            nbit = ys.shape[1]
            
            # whitening
            mean = ys.mean(dim=0, keepdim=True) # NxD -> 1xD
            ys_centered = ys - mean # NxD
            cov = ys_centered.T @ ys_centered
            e, v = torch.linalg.eigh(cov)
            L = torch.diag(1.0 / torch.pow(e, exponent=0.5))
            weight = torch.mm(L, v.T)
            bias = -torch.mm(mean, weight.T).squeeze(0)
            linear = nn.Linear(nbit, nbit, bias=True)
            linear.weight.data = np.sqrt(nbit) * weight
            linear.bias.data = np.sqrt(nbit) * bias
            msg_decoder = nn.Sequential(msg_decoder, linear.to(device))
            torchscript_m = torch.jit.script(msg_decoder)
            params.msg_decoder_path = params.msg_decoder_path.replace(".pth", "_whit.pth")
            print(f'>>> Creating torchscript at {params.msg_decoder_path}...')
            torch.jit.save(torchscript_m, params.msg_decoder_path)
    
    msg_decoder.eval()
    nbit = msg_decoder(torch.zeros(1, 3, 128, 128).to(device)).shape[-1]
    
    # freeze message decoder
    for param in [*msg_decoder.parameters()]:
        param.requires_grad = False
        
    # """
    
    # Freeze LDM
    for param in [*ldm_ae.parameters()]:
        param.requires_grad = False

    # Loads the data
    print(f'>>> Loading data from {params.train_dir} and {params.val_dir}...')
    vqgan_transform = transforms.Compose([
        transforms.Resize(params.img_size),
        transforms.CenterCrop(params.img_size),
        transforms.ToTensor(),
        utils_img.normalize_vqgan,
    ])
    train_loader = utils.get_dataloader(params.train_dir, vqgan_transform, params.batch_size, num_imgs=params.batch_size*params.steps, shuffle=True, num_workers=4, collate_fn=None)
    val_loader = utils.get_dataloader(params.val_dir, vqgan_transform, params.batch_size*4, num_imgs=1000, shuffle=False, num_workers=4, collate_fn=None)
    vqgan_to_imnet = transforms.Compose([utils_img.unnormalize_vqgan, utils_img.normalize_img])
    
    # Create losses
    print(f'>>> Creating losses...')
    print(f'Losses: {params.loss_w} and {params.loss_i}...')
    if params.loss_w == 'mse':        
        loss_w = lambda decoded, keys, temp=10.0: torch.mean((decoded*temp - (2*keys-1))**2) # b k - b k
    elif params.loss_w == 'bce':
        loss_w = lambda decoded, keys, temp=10.0: F.binary_cross_entropy_with_logits(decoded*temp, keys, reduction='mean')
    else:
        raise NotImplementedError
    
    if params.loss_i == 'mse':
        loss_i = lambda imgs_w, imgs: torch.mean((imgs_w - imgs)**2)
    elif params.loss_i == 'watson-dft':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-DFT', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif params.loss_i == 'watson-vgg':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    elif params.loss_i == 'ssim':
        provider = LossProvider()
        loss_percep = provider.get_loss_function('SSIM', colorspace='RGB', pretrained=True, reduction='sum')
        loss_percep = loss_percep.to(device)
        loss_i = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0)/ imgs_w.shape[0]
    else:
        raise NotImplementedError

    

    # Creating key
    print(f'\n>>> Creating key with {nbit} bits...')
    if params.key == "":
        key = torch.randint(0, 2, (1, nbit), dtype=torch.float32, device=device)
    else:
        key = torch.tensor([[int(key_i) for key_i in params.key]], dtype=torch.float32, device=device)
    key_str = "".join([ str(int(ii)) for ii in key.tolist()[0]])
    print(f'Key: {key_str}')

    # Copy the LDM decoder and finetune the copy
    ldm_decoder = deepcopy(ldm_ae)
    ldm_decoder.encoder = nn.Identity()
    ldm_decoder.quant_conv = nn.Identity()
    ldm_decoder.to(device)
    for param in ldm_decoder.parameters():
        param.requires_grad = True
    optim_params = utils.parse_params(params.optimizer)
    optimizer = utils.build_optimizer(model_params=ldm_decoder.parameters(), **optim_params)

    # Training loop
    print(f'>>> Training...')
            
    train_stats = train(train_loader, optimizer, loss_w, loss_i, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
    val_stats = val(val_loader, ldm_ae, ldm_decoder, msg_decoder, vqgan_to_imnet, key, params)
    log_stats = {'key': key_str,
            **{f'train_{k}': v for k, v in train_stats.items()},
            **{f'val_{k}': v for k, v in val_stats.items()},
        }
    save_dict = {
        'ldm_decoder': ldm_decoder.state_dict(),
        'optimizer': optimizer.state_dict(),
        'params': params,
    }

    # Save checkpoint
    torch.save(save_dict, os.path.join(params.output_dir, f"checkpoint.pth"))
    with (Path(params.output_dir) / "log.txt").open("a") as f:
        f.write(json.dumps(log_stats) + "\n")
    with (Path(params.output_dir) / "keys.txt").open("a") as f:
        f.write(os.path.join(params.output_dir, f"checkpoint.pth") + "\t" + key_str + "\n")
    print('\n')

def train(data_loader: Iterable, optimizer: torch.optim.Optimizer, loss_w: Callable, loss_i: Callable, ldm_ae: AutoencoderKL, ldm_decoder:AutoencoderKL, msg_decoder: nn.Module, vqgan_to_imnet:nn.Module, key: torch.Tensor, params: argparse.Namespace):
    header = 'Train'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.train()
    base_lr = optimizer.param_groups[0]["lr"]
    
    use_pretrained_discriminator = True
    
    # discriminator = NLayerDiscriminator(ndf=128, n_layers=4)
    discriminator = ResnetDiscriminator()
    # discriminator.load_state_dict(torch.load("blackbox_checkpoints/resnet_142/model.ckpt")["state_dict"])
    if use_pretrained_discriminator:
        discriminator.load_state_dict(torch.load("pretrained_discriminator.pth"))
    discriminator.to(device)
    optim_params = utils.parse_params(params.optimizer)
    optim_params["lr"] = 1e-4
    optimizer_D = utils.build_optimizer(model_params=discriminator.parameters(), **optim_params)
    discriminator.train()
    
    # pretrain discriminator
    discr_pretrain_steps = 2500
    
    if use_pretrained_discriminator:
        discr_pretrain_steps = 100
        
    if True:
        accloss, accacc = [], []
        for ii, imgs in enumerate(data_loader):
            imgs = imgs.to(device)
            with torch.no_grad():
                imgs_w = ldm_decoder.decode(ldm_ae.encode(imgs).mode()).detach()     # should be predicted as fake
            discr_pred_real = discriminator(vqgan_to_imnet(imgs))
            discr_pred_wm = discriminator(vqgan_to_imnet(imgs_w))
            
            loss = torch.log(torch.sigmoid(discr_pred_real)) + torch.log(1 - torch.sigmoid(discr_pred_wm))
            (-loss).mean().backward()
            optimizer_D.step()
            optimizer_D.zero_grad()
            
            accuracy = torch.cat([discr_pred_real > 0, discr_pred_wm < 0], 0).float()
            accacc.append(accuracy)
            accloss.append(loss)
            
            if ii % 100 == 0:
                accacc = torch.cat(accacc, 0)
                accloss = torch.cat(accloss, 0)
                print(f"step {ii}, discriminator loss: {accloss.mean().cpu().item():.3f}, accuracy: {accacc.mean().cpu().item()*100:.3f}%")
                accloss = []
                accacc = []
            if ii >= discr_pretrain_steps:
                break
        
    print("pretrained discriminator")
        
    discr_steps_per_gen_step = 5
    
    save_imgs = []
    count = 0
    for save_imgs_i in data_loader:      # 2 batches
        save_imgs.append(save_imgs_i.to(device))
        count += 1
        if count >= 2:
            break
        
    queue = [-10000]
    
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        imgs = imgs.to(device)
        keys = key.repeat(imgs.shape[0], 1)
        
        utils.adjust_learning_rate(optimizer, ii, params.steps, params.warmup_steps, base_lr)
        # encode images
        with torch.no_grad():
            imgs_z = ldm_ae.encode(imgs) # b c h w -> b z h/f w/f
            imgs_z = imgs_z.mode().detach()

        # decode latents with original and finetuned decoder
        # imgs_d0 = ldm_ae.decode(imgs_z) # b z h/f w/f -> b c h w
        imgs_d0 = imgs
        
        if np.mean(queue) < -0.5:
        # if ii % discr_steps_per_gen_step == 0:
            imgs_w = ldm_decoder.decode(imgs_z) # b z h/f w/f -> b c h w
        else:
            with torch.no_grad():
                imgs_w = ldm_decoder.decode(imgs_z).detach() # b z h/f w/f -> b c h w
                
        discr_grad_img = torch.nn.Parameter(torch.zeros_like(imgs_w[0:1]))
                
        discr_pred_real = discriminator(vqgan_to_imnet(imgs))
        discr_pred_wm = discriminator(vqgan_to_imnet(imgs_w + discr_grad_img))
        
        
        lossd = torch.log(torch.sigmoid(discr_pred_real)) + torch.log(1 - torch.sigmoid(discr_pred_wm))
        loss = lossd.mean() * params.lambda_d
        queue.append(lossd.mean().cpu().item())
        while len(queue) > 3:
            queue.pop(0)
        
        # # savedgrads = {k: v.grad.clone() for k, v in ldm_decoder.named_parameters()}
        # optimizer_D.zero_grad()     # removes only grads on discriminator, leaves the ones on generator
        # optimizer.zero_grad()
        
        if np.mean(queue) > -0.5 or ii == 0:
        # if ii % discr_steps_per_gen_step == 0:
            # UPDATE GENERATOR
            
            # # copy grads from generator
            # gen_grads_adv = {k: v.grad for k, v in ldm_decoder.named_parameters()}
            # # flip the grad because we want to minimize lossd next
            # gen_grads_adv = {k: -v for k, v in gen_grads_adv.items()}
            # # compute grad norm
            # gen_grad_adv_norm = sum([grad.sum() for grad in gen_grads_adv.values()])
            # gen_grad_adv_norm_count = sum([grad.numel() for grad in gen_grads_adv.values()])

            
            # compute loss
            perc_grad_img = torch.nn.Parameter(torch.zeros_like(imgs_w[0:1]))
            lossi = loss_i(imgs_w + perc_grad_img, imgs_d0)
            loss = loss + params.lambda_i * lossi
            
            decoded = msg_decoder(vqgan_to_imnet(imgs_w)) # b c h w -> b k
            lossw = loss_w(decoded, keys)
            # loss = loss + params.lambda_w * lossw

            # optim step
            loss.backward()                
            optimizer.step()
            optimizer.zero_grad()
        else:
            loss.backward()
            
        # flip and rescale back the gradient on discriminator
        for k, v in discriminator.named_parameters():
            v.grad = v.grad * (-1) / params.lambda_d
        optimizer_D.step()
        optimizer_D.zero_grad()

        # log stats
        diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
        bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
        # word_accs = (bit_accs == 1) # b
        log_stats = {
            "iteration": int(ii),
            "loss": loss.item(),
            "loss_w": lossw.item(),
            "loss_i": lossi.item(),
            "loss_d": lossd.mean().item(),
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
            "bit_acc_avg": torch.mean(bit_accs).item(),
            # "word_acc_avg": torch.mean(word_accs.type(torch.float)).item(),
            "lr": optimizer.param_groups[0]["lr"],
        }
        for _name, _loss in log_stats.items():
            metric_logger.update(**{_name:_loss})
        # if ii % params.log_freq == 0:
        #     print(json.dumps(log_stats))
        
        # save images during training
        if ii % params.save_img_freq == 0:
            save_imgs_d0 = []
            save_imgs_w = []
            for save_imgs_i in save_imgs:
                # encode images
                imgs_z_i = ldm_ae.encode(save_imgs_i).mode() # b c h w -> b z h/f w/f

                # decode latents with original and finetuned decoder
                # imgs_d0_i = ldm_ae.decode(imgs_z_i) # b z h/f w/f -> b c h w
                imgs_d0_i = save_imgs_i
                imgs_w_i = ldm_decoder.decode(imgs_z_i) # b z h/f w/f -> b c h w
                save_imgs_d0.append(imgs_d0_i)
                save_imgs_w.append(imgs_w_i)
            _save_imgs = torch.cat(save_imgs, 0)
            save_imgs_d0 = torch.cat(save_imgs_d0, 0)
            save_imgs_w = torch.cat(save_imgs_w, 0)
            
            save_image(torch.clamp(utils_img.unnormalize_vqgan(_save_imgs),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(save_imgs_d0),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(save_imgs_w),0,1), os.path.join(params.imgs_dir, f'{ii:03}_train_w.png'), nrow=8)
    
            save_image(torch.clamp(utils_img.unnormalize_vqgan(_save_imgs),0,1), os.path.join(params.imgs_dir, f'latest_train_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(save_imgs_d0),0,1), os.path.join(params.imgs_dir, f'latest_train_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(save_imgs_w),0,1), os.path.join(params.imgs_dir, f'latest_train_w.png'), nrow=8)
            
    print("Averaged {} stats:".format('train'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def val(data_loader: Iterable, ldm_ae: AutoencoderKL, ldm_decoder: AutoencoderKL, msg_decoder: nn.Module, vqgan_to_imnet:nn.Module, key: torch.Tensor, params: argparse.Namespace):
    header = 'Eval'
    metric_logger = utils.MetricLogger(delimiter="  ")
    ldm_decoder.decoder.eval()
    for ii, imgs in enumerate(metric_logger.log_every(data_loader, params.log_freq, header)):
        
        imgs = imgs.to(device)

        imgs_z = ldm_ae.encode(imgs) # b c h w -> b z h/f w/f
        imgs_z = imgs_z.mode()

        # imgs_d0 = ldm_ae.decode(imgs_z) # b z h/f w/f -> b c h w
        imgs_d0 = imgs
        imgs_w = ldm_decoder.decode(imgs_z) # b z h/f w/f -> b c h w
        
        keys = key.repeat(imgs.shape[0], 1)

        log_stats = {
            "iteration": ii,
            "psnr": utils_img.psnr(imgs_w, imgs_d0).mean().item(),
            # "psnr_ori": utils_img.psnr(imgs_w, imgs).mean().item(),
        }
        attacks = {
            'none': lambda x: x,
            'crop_01': lambda x: utils_img.center_crop(x, 0.1),
            'crop_05': lambda x: utils_img.center_crop(x, 0.5),
            'rot_25': lambda x: utils_img.rotate(x, 25),
            'rot_90': lambda x: utils_img.rotate(x, 90),
            'resize_03': lambda x: utils_img.resize(x, 0.3),
            'resize_07': lambda x: utils_img.resize(x, 0.7),
            'brightness_1p5': lambda x: utils_img.adjust_brightness(x, 1.5),
            'brightness_2': lambda x: utils_img.adjust_brightness(x, 2),
            'jpeg_80': lambda x: utils_img.jpeg_compress(x, 80),
            'jpeg_50': lambda x: utils_img.jpeg_compress(x, 50),
        }
        for name, attack in attacks.items():
            imgs_aug = attack(vqgan_to_imnet(imgs_w))
            decoded = msg_decoder(imgs_aug) # b c h w -> b k
            diff = (~torch.logical_xor(decoded>0, keys>0)) # b k -> b k
            bit_accs = torch.sum(diff, dim=-1) / diff.shape[-1] # b k -> b
            word_accs = (bit_accs == 1) # b
            log_stats[f'bit_acc_{name}'] = torch.mean(bit_accs).item()
            # log_stats[f'word_acc_{name}'] = torch.mean(word_accs.type(torch.float)).item()
        for name, loss in log_stats.items():
            metric_logger.update(**{name:loss})

        if ii % params.save_img_freq == 0:
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_orig.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_d0),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_d0.png'), nrow=8)
            save_image(torch.clamp(utils_img.unnormalize_vqgan(imgs_w),0,1), os.path.join(params.imgs_dir, f'{ii:03}_val_w.png'), nrow=8)
    
    print("Averaged {} stats:".format('eval'), metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # run experiment
    main(params)
