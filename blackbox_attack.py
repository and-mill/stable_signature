import time
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
import fire
import numpy as np
from pathlib import Path
import tqdm
import logging

from torchvision import transforms
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import lpips
from PIL import Image

from utils_img import normalize_img, unnormalize_img

from watermark_utils import load_model, decode_message, eval_message

from torch.autograd import Variable

from loss.loss_provider import LossProvider
from torch.utils.data import DataLoader

from perlin_noise import perlin_noise
from torch.nn.functional import interpolate

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


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
    def __init__(self, wmpath="coco_wm_1000", realpath="coco_original_1000", wm_fraction=1., filenames=None):
        # composite threshold: the closer to 0, the more clean image is used
        super().__init__()
        self.wmpath, self.realpath = Path(wmpath), (Path(realpath) if realpath is not None else None)
        self.wm_fraction = wm_fraction if realpath is not None else 1
        self.length = None
        self.filenames = filenames
        if self.filenames is None:
            self.filenames = self.process_filenames(wmpath, realpath)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ])
        
    @classmethod
    def create_datasets(cls, wmpath="coco_wm_1000", realpath="coco_original_1000", wm_fraction=1.):
        filenames = cls.process_filenames(wmpath, realpath)
        numtrain = round(len(filenames) * 0.9)
        trainnames, validnames = filenames[:numtrain], filenames[numtrain:]
        trainds = BlackBoxAttackDataset(wmpath=wmpath, realpath=realpath, wm_fraction=wm_fraction,
                                        filenames=trainnames)
        validds = BlackBoxAttackDataset(wmpath=wmpath, realpath=realpath, wm_fraction=wm_fraction,
                                        filenames=validnames)
        return trainds, validds
        
    @classmethod
    def process_filenames(cls, wmpath, realpath):
        wmpaths = list(Path(wmpath).glob("*"))
        if realpath is not None:
            realpaths = list(Path(realpath).glob("*"))
            print(len(wmpaths))
            assert len(set([x.stem for x in wmpaths]) - set([x.stem for  x in realpaths])) == 0
        filenames = sorted([x.name for x in wmpaths])
        return filenames
        
    def __len__(self):
        return len(self.filenames) * 2 if self.realpath is not None else len(self.filenames)
        
    def __getitem__(self, i):
        if i < len(self.filenames):
            imgpath = self.wmpath / self.filenames[i]
            img = Image.open(imgpath).convert("RGB")
            imgtensor = self.transform(img)
            if self.wm_fraction < 1:     # if equals 1, then use the entire watermarked image instead of patching it
                realimgpath = self.realpath / self.filenames[i]
                realimg = Image.open(realimgpath).convert("RGB")
                realimgtensor = self.transform(realimg)
                # select random mask and collage according to mask; use perlin noise
                noisemap = generate_random_rmap(img.size)
                values = noisemap.flatten().sort().values
                composite_threshold = values[round(len(values) * self.wm_fraction)]
                mixmask = (noisemap > composite_threshold).float()[None]       # is one where real image must be
                mixmask = transforms.functional.gaussian_blur(mixmask, kernel_size=(11, 11))
                imgtensor = mixmask * realimgtensor + (1 - mixmask) * imgtensor
            y = torch.ones(1,).to(imgtensor.device) * 1
        else:
            imgpath = self.realpath / self.filenames[i- len(self.filenames)]
            img = Image.open(imgpath).convert("RGB")
            imgtensor = self.transform(img)
            y = torch.ones(1,).to(imgtensor.device) * 0
        return imgtensor, y
    
    

class ConvBNRelu(torch.nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """
    def __init__(self, channels_in, channels_out):

        super(ConvBNRelu, self).__init__()
        
        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(channels_in, channels_out, 3, stride=1, padding=1),
            torch.nn.BatchNorm2d(channels_out, eps=1e-3),
            torch.nn.GELU()
        )

    def forward(self, x):
        return self.layers(x)


class HiddenDecoder(torch.nn.Module):
    """
    Decoder module. Receives a watermarked image and extracts the watermark.
    The input image may have various kinds of noise applied to it,
    such as Crop, JpegCompression, and so on. See Noise layers for more.
    """
    def __init__(self, num_blocks, num_bits, channels):

        super(HiddenDecoder, self).__init__()

        layers = [ConvBNRelu(3, channels)]
        for _ in range(num_blocks - 1):
            layers.append(ConvBNRelu(channels, channels))

        layers.append(ConvBNRelu(channels, num_bits))
        layers.append(torch.nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.layers = torch.nn.Sequential(*layers)

        self.linear = torch.nn.Linear(num_bits, num_bits)

    def forward(self, img_w):

        x = self.layers(img_w) # b d 1 1
        x = x.squeeze(-1).squeeze(-1) # b d
        x = self.linear(x) # b d
        return x
    
    
class LitClassifier(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        print("creating model")
        # self.model = HiddenDecoder(8, 1, 3)
        self.model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        self.model.fc = torch.nn.Linear(512, 1)
        print("model created")
        self.loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        self.lr = lr
        
    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"train_acc": acc, "loss": loss}
        self.log_dict(metrics)
        return metrics
    
    def training_epoch_end(self, outputs):
        ret = {}
        for output in outputs:
            if len(ret) == 0:
                for k, v in output.items():
                    ret[k] = []
            for k, v in output.items():
                ret[k].append(v.cpu().item())
        for k in ret:
            ret[k] = np.mean(ret[k])
        print(f"Training epoch metrics: {ret}")
    
    def validation_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"val_acc": acc, "val_loss": loss}
        self.log_dict(metrics)
        return metrics
    
    def validation_epoch_end(self, outputs) -> None:
        ret = {}
        for output in outputs:
            if len(ret) == 0:
                for k, v in output.items():
                    ret[k] = []
            for k, v in output.items():
                ret[k].append(v.cpu().item())
        for k in ret:
            ret[k] = np.mean(ret[k])
        print(f"Validation epoch metrics: {ret}")

    def test_step(self, batch, batch_idx):
        loss, acc = self._shared_eval_step(batch, batch_idx)
        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics)
        return metrics
        
    def _shared_eval_step(self, batch, batch_idx):
        x, y = batch
        pred = self.model(x)
        loss = self.loss(pred, y).mean()
        acc = ((pred > 0) == y).float().mean()
        return loss, acc
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        
    
def main(wmpath="images/coco_wm_1000/", realpath="images/coco_original_1000", lr=5e-4, batsize=16, device=0, seed=42, 
         debug=False):
    seed_everything(seed)
    VALID_BATSIZE = 8
    WM_FRACTION = 0.15
    
    numworkers = batsize
    if debug:
        numworkers = 0
        
    gpu = device
    device = torch.device("cuda", device)
    
    model = LitClassifier(lr=lr)
    
    # load data
    print("load data")
    trainds, validds = BlackBoxAttackDataset.create_datasets(wmpath, realpath, wm_fraction=WM_FRACTION)
    traindl = DataLoader(trainds, batch_size=batsize, shuffle=True, num_workers=numworkers)
    validdl = DataLoader(validds, batch_size=VALID_BATSIZE, shuffle=False, num_workers=VALID_BATSIZE)
    
    testds = BlackBoxAttackDataset(wmpath="images/generated", realpath=None)
    testdl = DataLoader(testds, batch_size=VALID_BATSIZE, shuffle=False, num_workers=VALID_BATSIZE)
    print(f"data loaded: {len(trainds)} train examples, {len(traindl)} batches, {len(validds)} test examples, {len(validdl)} batches")
    
    
    # msg_extractor = torch.jit.load("models/dec_48b_whit.torchscript.pt").to(device)
    
    # bitaccs, pvals = [], []
    # for spath in Path().glob(path):
    #     print(f"Doing file: {spath}")
    #     img = Image.open(spath)
        
    #     print("decoding message")
    #     # print(f"decoded message: {msg}")
    #     bitacc, pval = eval_message(msg)
    #     bitaccs.append(bitacc); pvals.append(pval)
        
    # print(f"Average bit acc.: {np.mean(bitaccs)}")
    
    # COMPUTES AVERAGE
    # bitaccs = []
    # labels = []
    # for batch in tqdm.tqdm(validdl):
    #     # run image through watermark decoder
    #     batch_msgs = decode_message(batch[0], msg_extractor=msg_extractor, device=device, verbose=False)
    #     batch_bitaccs = [eval_message(batch_msg, verbose=False)[0] for batch_msg in batch_msgs]
    #     bitaccs.extend(batch_bitaccs)
    #     labels.extend(batch[1].squeeze().cpu().numpy().tolist())
    # # print statistics of bit accs for train and valid data subsets
    # cleanbitaccs = [bitacc for (bitacc, label) in zip(bitaccs, labels) if label == 0]
    # dirtybitaccs = [bitacc for (bitacc, label) in zip(bitaccs, labels) if label == 1]
    # print(f"Average (on test set) clean bit acc for wm_fraction={WM_FRACTION}: {np.mean(cleanbitaccs)}, watermarked bit acc: {np.mean(dirtybitaccs)}")
            
    # test
    testaccs = []
    for batch in testdl:
        outputs = model.model(batch[0])
        acc = ((outputs.squeeze() > 0).float() == batch[1].squeeze()).float().cpu().numpy().tolist()
        testaccs.extend(acc)
    print(f"Test accuracy on generated watermarked images before training: {np.mean(testaccs):.3f}")
        
    print("train model")
    checkpointer = ModelCheckpoint(dirpath="blackbox_checkpoints/resnet_{seed}",
                                   save_weights_only=True, every_n_epochs=1, filename="{epoch}-{val_acc:.2f}")
    trainer = pl.Trainer(gpus=[gpu], max_epochs=3, callbacks=[checkpointer])
    trainer.fit(model, traindl, validdl)
    
    # test
    testaccs = []
    for batch in testdl:
        outputs = model.model(batch[0])
        acc = ((outputs.squeeze() > 0).float() == batch[1].squeeze()).float().cpu().numpy().tolist()
        testaccs.extend(acc)
    print(f"Test accuracy on generated watermarked images after training: {np.mean(testaccs):.3f}")
            
        
        
    
    
    

    


if __name__ == "__main__":
    fire.Fire(main)