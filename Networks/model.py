import os
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

base_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)
if True:
    from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    from Networks.dl_v3 import Seg


# REF: https://github.com/ChaoningZhang/MobileSAM
sam_checkpoint = os.path.abspath(os.path.join(
    base_folder, "./weights/mobile_sam.pt"
))
model_type = "vit_t"


class SegSAM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mobile_sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        mobile_sam.eval()
        mobile_sam.requires_grad_(False)

        self.mobile_sam = mobile_sam
        self.predictor = SamPredictor(self.mobile_sam)
        self.seg = torch.load(os.path.abspath(
            os.path.join(base_folder, "./Networks/Seg.model")
        ))
        pass

    def forward(self, input: torch.Tensor):
        batch_length = int(input.shape[0])
        # mask = F.interpolate(self.seg(input), size=(256, 256), mode="bilinear")
        mask = torch.ones((len(input), 1, 256, 256))
        output = []
        for i in range(batch_length):
            input_numpy = (input[i] * 255).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
            mask_numpy = (mask[i] * 255).to(torch.uint8).cpu().numpy()
            
            self.predictor.set_image(input_numpy)
            masks, _, _ = self.predictor.predict(mask_input=mask_numpy, multimask_output=True)
            masks = torch.from_numpy(masks)
            masks = (masks > self.mobile_sam.mask_threshold).to(input.device)
            masks = torch.max(masks.to(torch.float32), dim=0).values
            output.append(masks)
        output = torch.stack(output)
        return output


if __name__ == "__main__":
    segsam = SegSAM()
    from Data.Gaofen import train_loader, val_loader, test_loader, len_train, len_val, len_test
    for x, y, z in train_loader:
        pre = segsam(x)
        print(pre.to(torch.float).mean())
        print(pre.shape)
        pass
