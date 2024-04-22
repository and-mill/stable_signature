import time
from pytorch_lightning import seed_everything
import torch
import fire
import numpy as np
from pathlib import Path

from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from PIL import Image

from utils_img import normalize_img, unnormalize_img

from watermark_utils import load_model, decode_message_pil, eval_message

from torch.autograd import Variable

from loss.loss_provider import LossProvider
from torch.utils.data import DataLoader

from perlin_noise import perlin_noise
from torch.nn.functional import interpolate


def generate_random_rmap(shape=(512,512), gridsize=2, rescale=1):
    start = time.time()
    _shape = tuple([x//rescale for x in shape])
    noise = perlin_noise(grid_shape=(gridsize, gridsize), out_shape=_shape)
    end = time.time()
    noise = (noise - noise.min()) / (noise.max() - noise.min())
    noise = interpolate(noise[None, None], shape, mode="bilinear")[0, 0]
    return noise


class MessageLoss(torch.nn.Module):
    def __init__(self, bits=48):
        super().__init__()
        self.set_target_message(bits=bits)
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.msg_extractor = torch.jit.load("models/dec_48b_whit.torchscript.pt")
        
    def set_target_message(self, msg=None, bits=48):
        if msg is None:
            msg = (torch.randn(bits) > 0).float()
        self.register_buffer("target_msg", msg)
        
    def initialize_flip_message(self, img:torch.Tensor):
        with torch.no_grad():
            self.set_target_message((self.msg_extractor(img[None])[0] < 0.).float())
        
    def forward(self, img:torch.Tensor, msg_logits:torch.Tensor):
        img = img * 0.5 + 0.5
        img = normalize_img(img)
        loss = self.loss(msg_logits, self.target_msg[None]).sum(-1).mean()
        return loss


def adversarial_original_decoder_attack(pred, net="vgg", lr=1e-3, steps=40, evalinterval=5, 
                                        lpipsweight=1, mseweight=1, msgweight=0.5, device=torch.device("cpu")):
    
    def to_pil(_img):
        _img = _img[0]
        return transforms.ToPILImage()(_img * 0.5 + 0.5)
    
    lpips_loss = lpips.LPIPS(net=net).to(device)
    msg_loss = MessageLoss().to(device)
    mse = torch.nn.MSELoss().to(device)
    
    pred = transforms.ToTensor()(pred).to(device) * 2 - 1
    pred = pred[None]
    ref = torch.empty_like(pred).copy_(pred)
    pred = Variable(pred, requires_grad=True)
    
    
    optimizer = torch.optim.Adam([pred,], lr=lr, betas=(0.9, 0.999))

    for i in range(steps):
        print(f"Step {i}")
        if i % evalinterval == 0:
            # evaluate PSNR and watermark bits
            psnr = peak_signal_noise_ratio(np.asarray(to_pil(ref)), np.asarray(to_pil(pred)))
            print(f"PSNR: {psnr}")
            msg = decode_message_pil(to_pil(pred))
            eval_message(msg)
        
        optimizer.zero_grad()
        msg_logits = msg_loss.msg_extractor(pred)
        dist = lpipsweight * lpips_loss.forward(pred, ref) + msgweight * msg_loss(pred, msg_logits) + mseweight * mse(pred, ref)
        dist.backward()
        optimizer.step()
        pred.data = torch.clamp(pred.data, -1, 1)
        
    return to_pil(pred)


class BinaryEntropy(torch.nn.Module):
    def __init__(self, reduction="mean"):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, img:torch.Tensor, msg_logits:torch.Tensor):
        img = img * 0.5 + 0.5
        img = normalize_img(img)
        msg_probs = torch.sigmoid(msg_logits)
        
        assert msg_probs.min() >= 0 and msg_probs.max() <= 1
        entropy = torch.distributions.Bernoulli(msg_probs).entropy().mean()
        return - entropy


def adversarial_original_decoder_attack_entropy(pred, net="vgg", lr=1e-3, bitacc_threshold=0.62, maxsteps=200, evalinterval=10, 
                                        lpipsweight=1, mseweight=1, msgweight=4, entropyweight=0, device=torch.device("cpu")):
    
    def to_pil(_img):
        _img = _img[0]
        return transforms.ToPILImage()(_img * 0.5 + 0.5)
    
    # lpips_loss = lpips.LPIPS(net=net).to(device)
    provider = LossProvider()
    loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='none')
    loss_percep = loss_percep.to(device)
    lpips_loss = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0).mean()
    msg_loss = MessageLoss().to(device)
    entropy_loss = BinaryEntropy().to(device)
    mse = torch.nn.MSELoss().to(device)
    msg_extractor = torch.jit.load("models/dec_48b_whit.torchscript.pt").to(device)
    
    pred = transforms.ToTensor()(pred).to(device) * 2 - 1
    pred = pred[None]
    ref = torch.empty_like(pred).copy_(pred)
    pred = Variable(pred, requires_grad=True)
    
    
    optimizer = torch.optim.Adam([pred,], lr=lr, betas=(0.9, 0.999))
    
    with torch.no_grad():
        entropy_before = entropy_loss(pred, msg_loss.msg_extractor(pred))

    for i in range(maxsteps):
        msg = decode_message_pil(to_pil(pred), msg_extractor=msg_extractor, device=device, verbose=False)
        bitacc, _ = eval_message(msg, verbose=False)
        if bitacc < bitacc_threshold:
            break
        if i % evalinterval == 0:
            print(f"Step {i}")
            # evaluate PSNR and watermark bits
            psnr = peak_signal_noise_ratio(np.asarray(to_pil(ref)), np.asarray(to_pil(pred)))
            print(f"PSNR: {psnr:.4f}, Bit acc: {bitacc:.3f}")
        
        optimizer.zero_grad()
        msg_logits = msg_loss.msg_extractor(pred)
        dist = lpipsweight * lpips_loss(pred, ref) + entropyweight * entropy_loss(pred, msg_logits) \
            + mseweight * mse(pred, ref) + msgweight * msg_loss(pred, msg_logits)
        dist.backward()
        optimizer.step()
        pred.data = torch.clamp(pred.data, -1, 1)
        
    with torch.no_grad():
        entropy_after = entropy_loss(pred, msg_loss.msg_extractor(pred))
        
    print(f"Entropies before and after: {entropy_before.cpu().numpy():.3f} --> {entropy_after.cpu().numpy():.3f}")
        
    return to_pil(pred)



def adversarial_original_decoder_attack_flip(pred, net="vgg", lr=1e-3, bitacc_threshold=0.60, maxsteps=200, evalinterval=10, 
                                        lpipsweight=1, mseweight=1, msgweight=4, entropyweight=0, device=torch.device("cpu")):
    
    def to_pil(_img):
        _img = _img[0]
        return transforms.ToPILImage()(_img * 0.5 + 0.5)
    
    # lpips_loss = lpips.LPIPS(net=net).to(device)
    provider = LossProvider()
    loss_percep = provider.get_loss_function('Watson-VGG', colorspace='RGB', pretrained=True, reduction='none')
    loss_percep = loss_percep.to(device)
    lpips_loss = lambda imgs_w, imgs: loss_percep((1+imgs_w)/2.0, (1+imgs)/2.0).mean()
    msg_loss = MessageLoss().to(device)
    entropy_loss = BinaryEntropy().to(device)
    mse = torch.nn.MSELoss().to(device)
    msg_extractor = torch.jit.load("models/dec_48b_whit.torchscript.pt").to(device)
    
    pred = transforms.ToTensor()(pred).to(device) * 2 - 1
    pred = pred[None]
    ref = torch.empty_like(pred).copy_(pred)
    pred = Variable(pred, requires_grad=True)
    
    msg_loss.initialize_flip_message(pred[0])
    print(msg_loss.target_msg)
    
    optimizer = torch.optim.Adam([pred,], lr=lr, betas=(0.9, 0.999))
    
    with torch.no_grad():
        entropy_before = entropy_loss(pred, msg_loss.msg_extractor(pred))

    for i in range(maxsteps):
        msg = decode_message_pil(to_pil(pred), msg_extractor=msg_extractor, device=device, verbose=False)
        bitacc, _ = eval_message(msg, verbose=False)
        if bitacc < bitacc_threshold:
            break
        if i % evalinterval == 0:
            print(f"Step {i}")
            # evaluate PSNR and watermark bits
            psnr = peak_signal_noise_ratio(np.asarray(to_pil(ref)), np.asarray(to_pil(pred)))
            print(f"PSNR: {psnr:.4f}, Bit acc: {bitacc:.3f}")
        
        optimizer.zero_grad()
        msg_logits = msg_loss.msg_extractor(pred)
        dist = lpipsweight * lpips_loss(pred, ref) + entropyweight * entropy_loss(pred, msg_logits) \
            + mseweight * mse(pred, ref) + msgweight * msg_loss(pred, msg_logits)
        dist.backward()
        optimizer.step()
        pred.data = torch.clamp(pred.data, -1, 1)
        
    with torch.no_grad():
        entropy_after = entropy_loss(pred, msg_loss.msg_extractor(pred))
        
    print(f"Entropies before and after: {entropy_before.cpu().numpy():.3f} --> {entropy_after.cpu().numpy():.3f}")
        
    return to_pil(pred)


def main_(device=1, path="generated/*.wm.png", seed=None, outputdir="attacked_wm_flip"):
    device = torch.device("cuda", device)
    
    if seed is not None:
        seed_everything(seed)
    
    # print("loading model")
    # pipe = load_model(device)
    # print("loaded")
    
    bitaccs, pvals, psnrs = [], [], []
    for spath in Path().glob(path):
        print(f"Doing file: {spath}")
        img = Image.open(spath)
        cleaned = adversarial_original_decoder_attack_flip(img, device=device)
        
        psnr = peak_signal_noise_ratio(np.asarray(img), np.asarray(cleaned))
        ssim = structural_similarity(np.asarray(img), np.asarray(cleaned), channel_axis=2)
        print(f"PSNR: {psnr}, SSIM: {ssim}")
        psnrs.append(psnr)
        
        print("decoding message")
        msg = decode_message_pil(cleaned)
        # print(f"decoded message: {msg}")
        bitacc, pval = eval_message(msg)
        bitaccs.append(bitacc); pvals.append(pval)
        
        if outputdir is not None:
            outputpath = Path(outputdir) / spath.name
            if not outputpath.parent.exists():
                Path.mkdir(outputpath.parent, parents=True, exist_ok=False)
            cleaned.save(outputpath)
        
    print(f"Average bit acc.: {np.mean(bitaccs)}")
    print(f"Average PSNR: {np.mean(psnrs)}")
    
    
class BlackBoxAttackDataset:
    def __init__(self, wmpath="coco_wm_1000", realpath="coco_original_1000", composite_threshold=1.):
        super().__init__()
        self.wmpath, self.realpath = Path(wmpath), Path(realpath)
        self.composite_threshold = composite_threshold
        self.length = None
        self.initialize()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        
    def initialize(self):
        wmpaths = list(Path(self.wmpath).glob("*"))
        realpaths = list(Path(self.realpath).glob("*"))
        print(len(wmpaths))
        assert len(set([x.stem for x in wmpaths]) - set([x.stem for  x in realpaths])) == 0
        self.filenames = sorted([x.name for x in wmpaths])
        
    def __len__(self):
        return len(self.filenames) * 2
        
    def __getitem__(self, i):
        if i >= len(self.filenames):
            imgpath = self.wmpath / self.filenames[i - len(self.filenames)]
            img = Image.open(imgpath).convert("RGB")
            imgtensor = self.transform(img)
            if self.composite_threshold < 1:     # if equals 1, then use the entire watermarked image instead of patching it
                realimgpath = self.realpath / self.filenames[i - len(self.filenames)]
                realimg = Image.open(realimgpath).convert("RGB")
                realimgtensor = self.transform(realimg)
                # select random mask and collage according to mask; use perlin noise
                noisemap = generate_random_rmap(img.size)
                mixmask = (noisemap > self.composite_threshold).float()[None]       # is one where real image must be
                mixmask = transforms.functional.gaussian_blur(mixmask, kernel_size=(11, 11))
                imgtensor = mixmask * realimgtensor + (1 - mixmask) * imgtensor
            y = torch.ones(1,).to(imgtensor.device) * 1
        else:
            imgpath = self.realpath / self.filenames[i]
            img = Image.open(imgpath).convert("RGB")
            imgtensor = self.transform(img)
            y = torch.ones(1,).to(imgtensor.device) * 0
        return imgtensor, y
        
    
def load_data(wmpath="coco_wm_1000", realpath="coco_original_1000", composite_threshold=1.):
    # TODO: split in train/valid and return two datasets
    pass
    
    
def main(wmpath="images/coco_wm_1000/", realpath="images/coco_original_1000", lr=1e-3, batsize=4):
    print("creating model")
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.fc = torch.nn.Linear(512, 1)
    print("model created")
    
    # load data
    print("load data")
    ds = BlackBoxAttackDataset(wmpath, realpath, composite_threshold=0.5)
    dl = DataLoader(ds, batch_size=batsize, shuffle=True, num_workers=batsize+1)
    print(f"data loaded: {len(ds)} examples, {len(dl)} batches")
    
    # TODO train binary classifier to predict which is watermarked and which is original
    for batch in dl:
        print(batch)
        
        
    
    
    

    


if __name__ == "__main__":
    fire.Fire(main)