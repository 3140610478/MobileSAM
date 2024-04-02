import os
import sys
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large, DeepLabV3_MobileNet_V3_Large_Weights
from torchvision.models.segmentation.deeplabv3 import FCNHead
from torchvision.models._utils import IntermediateLayerGetter

from functools import partial
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig


class Seg(nn.Module):
    def __init__(self):
        super().__init__()
        self.dl_v3 = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
        )
        self.conv = nn.Conv2d(42, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: Tensor) -> Tensor:
        y = self.dl_v3.forward(input)
        y_out, y_aux = y['out'], y['aux']
        y = torch.cat((y_out, y_aux), dim=1)
        return self.sigmoid(self.conv(y))

    def fit(self):
        self.conv.train()
        self.sigmoid.train()
        self.dl_v3.eval()
        self.dl_v3.requires_grad_(False)

    def train(self, *args, **kwargs):
        self.requires_grad_(True)
        super().train(*args, **kwargs)


class SegMultiscale(nn.Module):
    @staticmethod
    def _get_large_scale_classifier():
        bneck_conf = partial(InvertedResidualConfig, width_mult=1.0)
        classifier = nn.Sequential(
            InvertedResidual(
                bneck_conf(16, 3, 64, 16, False, "RE", 1, 1),
                partial(nn.BatchNorm2d, eps=0.001, momentum=0.01),
            ),
            FCNHead(16,  12),
        )
        return classifier

    def __init__(self):
        super().__init__()
        dl_v3 = deeplabv3_mobilenet_v3_large(
            weights=DeepLabV3_MobileNet_V3_Large_Weights.DEFAULT,
        )
        self.backbone: IntermediateLayerGetter = dl_v3.backbone
        self.backbone.return_layers = {
            "16": "0",
            "4": "1",
            # "3": "2",
            # "1": "3",
            "1": "2",
        }
        """
        # def forward(self, x):
        #     out = OrderedDict()
        #     for name, module in self.items():
        #         x = module(x)
        #         if name in self.return_layers:
        #             out_name = self.return_layers[name]
        #             out[out_name] = x
        #     return out
        # backbone.items()
        # 0  torch.Size([1, 16, 512, 512])
        # 1  torch.Size([1, 16, 512, 512])
        # 2  torch.Size([1, 24, 256, 256])
        # 3  torch.Size([1, 24, 256, 256])
        # 4  torch.Size([1, 40, 128, 128])
        # 5  torch.Size([1, 40, 128, 128])
        # 6  torch.Size([1, 40, 128, 128])
        # 7  torch.Size([1, 80, 64, 64])
        # 8  torch.Size([1, 80, 64, 64])
        # 9  torch.Size([1, 80, 64, 64])
        # 10 torch.Size([1, 80, 64, 64])
        # 11 torch.Size([1, 112, 64, 64])
        # 12 torch.Size([1, 112, 64, 64])
        # 13 torch.Size([1, 160, 64, 64])
        # 14 torch.Size([1, 160, 64, 64])
        # 15 torch.Size([1, 160, 64, 64])
        # 16 torch.Size([1, 960, 64, 64])

        in_channels = (960, 40, 48, 32, 12,)
        out_channels = (21, 21, 32, 24, 6,)
        input_shape = (64, 128, 256, 512, 1024)
        """

        classifier = (
            dl_v3.classifier,
            dl_v3.aux_classifier,
            # FCNHead(48, 16),
            # FCNHead(32,  9),
            SegMultiscale._get_large_scale_classifier(),
            nn.Conv2d(54, 1, 1),
        )
        upscale = (
            # nn.ConvTranspose2d(21, 12, 1, 4, output_padding=3),
            # nn.ConvTranspose2d(21, 12, 1, 2, output_padding=1),
            # nn.ConvTranspose2d(16, 16, 1, 2, output_padding=1, groups=16),
            # nn.ConvTranspose2d( 9,  9, 1, 2, output_padding=1, groups= 9),
        )
        self.classifier = nn.ModuleList(classifier)
        self.upscale = nn.ModuleList(upscale)
        self.sigmoid = nn.Sigmoid()
        pass

    def forward(self, x: Tensor) -> Tensor:
        input_shape = x.shape[-2:]
        features = self.backbone(x)

        # features['0'] = self.classifier[0](features['0'])
        # features['0'] = self.upscale[0](features['0'])

        # features['1'] = self.classifier[1](features['1'])
        # features['1'] = self.upscale[1](features['1'])

        # features['2'] = torch.cat(
        #     (features['2'], features['1'], features['0']), dim=1)
        # features['2'] = self.classifier[2](features['2'])
        # features['2'] = self.upscale[2](features['2'])

        # features['3'] = torch.cat(
        #     (features['3'], features['2']), dim=1)
        # features['3'] = self.classifier[3](features['3'])
        # features['3'] = self.upscale[3](features['3'])

        # x = torch.cat(
        #     (x, features['3']), dim=1)
        # x = self.classifier[4](x)
        # x = self.sigmoid(x)
        x0 = self.classifier[0](features['0'])
        x1 = self.classifier[1](features['1'])
        x2 = self.classifier[2](features['2'])
        x0 = F.interpolate(x0, size=input_shape,
                           mode="bilinear", align_corners=False)
        x1 = F.interpolate(x1, size=input_shape,
                           mode="bilinear", align_corners=False)
        x2 = F.interpolate(x2, size=input_shape,
                           mode="bilinear", align_corners=False)
        x = torch.cat((x2, x1, x0,), dim=1)
        x = self.classifier[-1](x)
        x = self.sigmoid(x)
        return x

    def fit(self):
        self.train()
        self.backbone.eval()
        self.backbone.requires_grad_(False)
        self.classifier[0].eval()
        self.classifier[0].requires_grad_(False)
        self.classifier[1].eval()
        self.classifier[0].requires_grad_(False)

    def train(self, *args, **kwargs):
        self.requires_grad_(True)
        super().train(*args, **kwargs)


if __name__ == "__main__":
    os.chdir(os.path.abspath(os.path.dirname(__file__)))
    seg = SegMultiscale()
    seg.fit()
    x = torch.zeros((1, 3, 1024, 1024))
    y = seg(x)
    pass
