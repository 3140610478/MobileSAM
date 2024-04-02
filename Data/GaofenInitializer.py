import os
import sys
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader, random_split
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)

data_folder = os.path.abspath(os.path.join(base_folder, "./Data/train_set"))
train_folder = os.path.abspath(os.path.join(base_folder, "./Data/train"))
if not os.path.exists(train_folder):
    os.mkdir(train_folder)
val_folder = os.path.abspath(os.path.join(base_folder, "./Data/val"))
if not os.path.exists(val_folder):
    os.mkdir(val_folder)
splitting = (1800, 200)


class DatasetTransformer(Dataset):
    def __init__(self, dataset: Dataset, transform=None, target_transform=None) -> None:
        self.dataset = dataset
        self.transform = transform
        self.target_transform = None

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.dataset[index]
        if self.transform is not None:
            x = self.transform(x)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return x, y

    def __len__(self):
        return len(self.dataset)


class GaofenInitializer(Dataset):
    def __init__(self, data_folder):
        self.data_folder = os.path.abspath(data_folder)
        self.image, self.label, self.roi = [
            os.path.abspath(os.path.join(self.data_folder, "./Image")),
            os.path.abspath(os.path.join(self.data_folder, "./Label")),
            os.path.abspath(os.path.join(self.data_folder, "./Roi")),
        ]
        self.ToTensor = transforms.ToTensor()
        self.length = len(os.listdir(self.image))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        if index >= self.length:
            raise StopIteration
        index += 1
        file_name = f"{index}.png"
        files = [
            os.path.abspath(os.path.join(
                self.data_folder, f"./Image/{file_name}")),
            os.path.abspath(os.path.join(
                self.data_folder, f"./Label/{file_name}")),
            os.path.abspath(os.path.join(
                self.data_folder, f"./Roi/{file_name}")),
        ]
        image, label, roi = [Image.open(file) for file in files]

        # return self.ToTensor(image), self.ToTensor(label), self.ToTensor(roi)
        return image, label, roi


train_set, val_set = random_split(
    GaofenInitializer(data_folder),
    splitting,
    torch.Generator().manual_seed(0),
)
# train_set = DatasetTransformer(train_set, train_transform)
# val_set = DatasetTransformer(val_set, val_transform)

if __name__ == "__main__":
    transform = transforms.ToTensor()
    inverse = transforms.ToPILImage()
    # for (x, y) in train_loader:
    #     print(x.shape)
    #     print(y.shape)
    #     pass
    i, l, r = "./image", "./label", "./roi"
    for p in (i, l, r):
        if not os.path.exists(os.path.join(train_folder, p)):
            os.mkdir(os.path.join(train_folder, p))
        if not os.path.exists(os.path.join(val_folder, p)):
            os.mkdir(os.path.join(val_folder, p))
    for index in range(len(train_set)):
        i, l, r = train_set[index]
        index += 1
        i.save(os.path.abspath(os.path.join(
            train_folder, f"./image/{index}.png")))
        l.save(os.path.abspath(os.path.join(
            train_folder, f"./label/{index}.png")))
        r.save(os.path.abspath(os.path.join(
            train_folder, f"./roi/{index}.png")))
    for index in range(len(val_set)):
        i, l, r = val_set[index]
        index += 1
        i.save(os.path.abspath(os.path.join(
            val_folder, f"./image/{index}.png")))
        l.save(os.path.abspath(os.path.join(
            val_folder, f"./label/{index}.png")))
        r.save(os.path.abspath(os.path.join(
            val_folder, f"./roi/{index}.png")))
