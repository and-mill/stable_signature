import numpy as np
import torch.nn as nn
import torch

class ConvRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=1, init_zero=False):
        super(ConvRelu, self).__init__()

        self.init_zero = init_zero
        if self.init_zero:
            self.layers = nn.Conv2d(channels_in, channels_out, 3, stride, padding=1)

        else:
            self.layers = nn.Sequential(
                nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
                nn.LeakyReLU(inplace=True)
            )

    def forward(self, x):
        return self.layers(x)

class ConvTRelu(nn.Module):
    def __init__(self, channels_in, channels_out, stride=2):
        super(ConvTRelu, self).__init__()
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(channels_in, channels_out, kernel_size=2, stride=stride, padding=0),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class ExpandNet(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(ExpandNet, self).__init__()

        layers = [ConvTRelu(in_channels, out_channels)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = ConvTRelu(out_channels, out_channels)
            layers.append(layer)

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

class Encoder(nn.Module):
    def __init__(self, H=128, message_length=64, channels=32):
        super(Encoder, self).__init__()

        stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))

        self.message_pre_layer = nn.Sequential(
            ConvRelu(1, channels),
            ExpandNet(channels, channels, blocks=stride_blocks),
            ConvRelu(channels, 1, init_zero=True),
        )

    def forward(self, message):
        size = int(np.sqrt(message.shape[1]))
        message_image = message.view(-1, 1, size, size)
        message = self.message_pre_layer(message_image)
        return message

class DecodeNet(nn.Module):
    def __init__(self, in_channels, out_channels, blocks):
        super(DecodeNet, self).__init__()

        layers = [ConvRelu(in_channels, out_channels, 2)] if blocks != 0 else []
        for _ in range(blocks - 1):
            layer = ConvRelu(out_channels, out_channels, 2)
            layers.append(layer)
        self.layers = nn.Sequential(*layers)
    def forward(self, x):
        return self.layers(x)

class Decoder(nn.Module):
    def __init__(self, H=128, message_length=64, channels=32):
        super(Decoder, self).__init__()
        stride_blocks = int(np.log2(H // int(np.sqrt(message_length))))
        self.message_layer = nn.Sequential(
            ConvRelu(3, channels),
            DecodeNet(channels, channels, blocks=stride_blocks),
            ConvRelu(channels, 1, init_zero=True),
        )

    def forward(self, message_image):
        message = self.message_layer(message_image)
        message = message.view(message.shape[0], -1)
        return message


class INN_block(nn.Module):
    def __init__(self, clamp=2.0):
        super().__init__()
        self.clamp = clamp
        self.r = Decoder()
        self.y = Decoder()
        self.f = Encoder()
        
    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x, rev=False):
        x1, x2 = x[0], x[1]
        if not rev:
            t2 = self.f(x2)
            y1 = x1 + t2
            s1, t1 = self.r(y1), self.y(y1)
            y2 = torch.exp(s1) * x2 + t1
            out = [y1, y2]
        else:
            s1, t1 = self.r(x1), self.y(x1)
            y2 = (x2 - t1) / torch.exp(s1)
            t2 = self.f(y2)
            y1 = (x1 - t2)
            out = [y1, y2]
        return out


class INN(nn.Module):
    def __init__(self, diff=False,length=64):
        super(INN, self).__init__()
        self.diff = diff
        self.inv1 = INN_block()
        self.inv2 = INN_block()
        self.inv3 = INN_block()
        self.inv4 = INN_block()
        self.inv5 = INN_block()
        self.inv6 = INN_block()
        self.inv7 = INN_block()
        self.inv8 = INN_block()
        if self.diff:
            self.diff_layer_pre = nn.Linear(length, 64)
            self.leakrelu = nn.LeakyReLU(inplace=True)
            self.diff_layer_post = nn.Linear(64, length)

    def forward(self, x, rev=False):
        if not rev:
            if self.diff:
                x1, x2 = x[0], x[1]
                x2 = self.leakrelu(self.diff_layer_pre(x2))
                x = [x1, x2]
            out = self.inv1(x)
            out = self.inv2(out)
            out = self.inv3(out)
            out = self.inv4(out)
            out = self.inv5(out)
            out = self.inv6(out)
            out = self.inv7(out)
            out = self.inv8(out)
        else:
            out = self.inv8(x, rev=True)
            out = self.inv7(out, rev=True)
            out = self.inv6(out, rev=True)
            out = self.inv5(out, rev=True)
            out = self.inv4(out, rev=True)
            out = self.inv3(out, rev=True)
            out = self.inv2(out, rev=True)
            out = self.inv1(out, rev=True)
            if self.diff:
                x1, x2 = out
                x2 = self.diff_layer_post(x2)
                out = [x1, x2]
        return out

class FED(nn.Module):
    def __init__(self, diff=False,length=64):
        super(FED, self).__init__()
        self.model = INN(diff, length)

    def forward(self, x, rev=False):

        if not rev:
            out = self.model(x)
        else:
            out = self.model(x, rev=True)
        return out
    
def clamp(input, min=None, max=None):
    ndim = input.ndimension()
    if min is None:
        pass
    elif isinstance(min, (float, int)):
        input = torch.clamp(input, min=min)
    elif isinstance(min, torch.Tensor):
        if min.ndimension() == ndim - 1 and min.shape == input.shape[1:]:
            input = torch.max(input, min.view(1, *min.shape))
        else:
            assert min.shape == input.shape
            input = torch.max(input, min)
    else:
        raise ValueError("min can only be None | float | torch.Tensor")

    if max is None:
        pass
    elif isinstance(max, (float, int)):
        input = torch.clamp(input, max=max)
    elif isinstance(max, torch.Tensor):
        if max.ndimension() == ndim - 1 and max.shape == input.shape[1:]:
            input = torch.min(input, max.view(1, *max.shape))
        else:
            assert max.shape == input.shape
            input = torch.min(input, max)
    else:
        raise ValueError("max can only be None | float | torch.Tensor")
    return input


class JpegBasic(nn.Module):
    def __init__(self):
        super(JpegBasic, self).__init__()

    def std_quantization(self, image_yuv_dct, scale_factor, round_func=torch.round):

        luminance_quant_tbl = (torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
            image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

        chrominance_quant_tbl = (torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float) * scale_factor).round().to(image_yuv_dct.device).clamp(min=1).repeat(
            image_yuv_dct.shape[2] // 8, image_yuv_dct.shape[3] // 8)

        q_image_yuv_dct = image_yuv_dct.clone()
        q_image_yuv_dct[:, :1, :, :] = image_yuv_dct[:, :1, :, :] / luminance_quant_tbl
        q_image_yuv_dct[:, 1:, :, :] = image_yuv_dct[:, 1:, :, :] / chrominance_quant_tbl
        q_image_yuv_dct_round = round_func(q_image_yuv_dct)
        return q_image_yuv_dct_round

    def std_reverse_quantization(self, q_image_yuv_dct, scale_factor):

        luminance_quant_tbl = (torch.tensor([
            [16, 11, 10, 16, 24, 40, 51, 61],
            [12, 12, 14, 19, 26, 58, 60, 55],
            [14, 13, 16, 24, 40, 57, 69, 56],
            [14, 17, 22, 29, 51, 87, 80, 62],
            [18, 22, 37, 56, 68, 109, 103, 77],
            [24, 35, 55, 64, 81, 104, 113, 92],
            [49, 64, 78, 87, 103, 121, 120, 101],
            [72, 92, 95, 98, 112, 100, 103, 99]
        ], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
            q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

        chrominance_quant_tbl = (torch.tensor([
            [17, 18, 24, 47, 99, 99, 99, 99],
            [18, 21, 26, 66, 99, 99, 99, 99],
            [24, 26, 56, 99, 99, 99, 99, 99],
            [47, 66, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99],
            [99, 99, 99, 99, 99, 99, 99, 99]
        ], dtype=torch.float) * scale_factor).round().to(q_image_yuv_dct.device).clamp(min=1).repeat(
            q_image_yuv_dct.shape[2] // 8, q_image_yuv_dct.shape[3] // 8)

        image_yuv_dct = q_image_yuv_dct.clone()
        image_yuv_dct[:, :1, :, :] = q_image_yuv_dct[:, :1, :, :] * luminance_quant_tbl
        image_yuv_dct[:, 1:, :, :] = q_image_yuv_dct[:, 1:, :, :] * chrominance_quant_tbl
        return image_yuv_dct

    def dct(self, image):
        coff = torch.zeros((8, 8), dtype=torch.float).to(image.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image.shape[2] // 8
        image_dct = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
        image_dct = torch.matmul(coff, image_dct)
        image_dct = torch.matmul(image_dct, coff.permute(1, 0))
        image_dct = torch.cat(torch.cat(image_dct.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image_dct

    def idct(self, image_dct):
        coff = torch.zeros((8, 8), dtype=torch.float).to(image_dct.device)
        coff[0, :] = 1 * np.sqrt(1 / 8)
        for i in range(1, 8):
            for j in range(8):
                coff[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * 8)) * np.sqrt(2 / 8)

        split_num = image_dct.shape[2] // 8
        image = torch.cat(torch.cat(image_dct.split(8, 2), 0).split(8, 3), 0)
        image = torch.matmul(coff.permute(1, 0), image)
        image = torch.matmul(image, coff)
        image = torch.cat(torch.cat(image.chunk(split_num, 0), 3).chunk(split_num, 0), 2)

        return image

    def rgb2yuv(self, image_rgb):
        image_yuv = torch.empty_like(image_rgb)
        image_yuv[:, 0:1, :, :] = 0.299 * image_rgb[:, 0:1, :, :] \
                                  + 0.587 * image_rgb[:, 1:2, :, :] + 0.114 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 1:2, :, :] = -0.1687 * image_rgb[:, 0:1, :, :] \
                                  - 0.3313 * image_rgb[:, 1:2, :, :] + 0.5 * image_rgb[:, 2:3, :, :]
        image_yuv[:, 2:3, :, :] = 0.5 * image_rgb[:, 0:1, :, :] \
                                  - 0.4187 * image_rgb[:, 1:2, :, :] - 0.0813 * image_rgb[:, 2:3, :, :]
        return image_yuv

    def yuv2rgb(self, image_yuv):
        image_rgb = torch.empty_like(image_yuv)
        image_rgb[:, 0:1, :, :] = image_yuv[:, 0:1, :, :] + 1.40198758 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 1:2, :, :] = image_yuv[:, 0:1, :, :] - 0.344113281 * image_yuv[:, 1:2, :, :] \
                                  - 0.714103821 * image_yuv[:, 2:3, :, :]
        image_rgb[:, 2:3, :, :] = image_yuv[:, 0:1, :, :] + 1.77197812 * image_yuv[:, 1:2, :, :]
        return image_rgb

    def yuv_dct(self, image, subsample):
        image = (clamp(image, -1, 1) + 1) * 255 / 2

        pad_height = (8 - image.shape[2] % 8) % 8
        pad_width = (8 - image.shape[3] % 8) % 8
        image = nn.ZeroPad2d((0, pad_width, 0, pad_height))(image)

        image_yuv = self.rgb2yuv(image)

        assert image_yuv.shape[2] % 8 == 0
        assert image_yuv.shape[3] % 8 == 0

        image_subsample = self.subsampling(image_yuv, subsample)

        image_dct = self.dct(image_subsample)

        return image_dct, pad_width, pad_height

    def idct_rgb(self, image_quantization, pad_width, pad_height):
        image_idct = self.idct(image_quantization)

        image_ret_padded = self.yuv2rgb(image_idct)

        image_rgb = image_ret_padded[:, :, :image_ret_padded.shape[2] - pad_height,
                    :image_ret_padded.shape[3] - pad_width].clone()

        image_rgb = image_rgb * 2 / 255 - 1
        return clamp(image_rgb, -1, 1)

    def subsampling(self, image, subsample):
        if subsample == 2:
            split_num = image.shape[2] // 8
            image_block = torch.cat(torch.cat(image.split(8, 2), 0).split(8, 3), 0)
            for i in range(8):
                if i % 2 == 1: image_block[:, 1:3, i, :] = image_block[:, 1:3, i - 1, :]
            for j in range(8):
                if j % 2 == 1: image_block[:, 1:3, :, j] = image_block[:, 1:3, :, j - 1]
            image = torch.cat(torch.cat(image_block.chunk(split_num, 0), 3).chunk(split_num, 0), 2)
        return image


class Jpeg(JpegBasic):
    def __init__(self, Q, subsample=0):
        super(Jpeg, self).__init__()

        self.Q = Q
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

        self.subsample = subsample

    def forward(self, image):
        image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

        image_quantization = self.std_quantization(image_dct, self.scale_factor)

        image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

        noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
        return noised_image


class JpegSS(JpegBasic):
    def __init__(self, Q, subsample=0):
        super(JpegSS, self).__init__()

        self.Q = Q
        self.scale_factor = 2 - self.Q * 0.02 if self.Q >= 50 else 50 / self.Q

        self.subsample = subsample

    def round_ss(self, x):
        cond = torch.tensor((torch.abs(x) < 0.5), dtype=torch.float).to(x.device)
        return cond * (x ** 3) + (1 - cond) * x

    def forward(self, image):
        image_dct, pad_width, pad_height = self.yuv_dct(image, self.subsample)

        image_quantization = self.std_quantization(image_dct, self.scale_factor, self.round_ss)

        image_quantization = self.std_reverse_quantization(image_quantization, self.scale_factor)

        noised_image = self.idct_rgb(image_quantization, pad_width, pad_height)
        return noised_image