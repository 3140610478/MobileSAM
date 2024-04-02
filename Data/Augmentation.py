import os
import sys
from PIL import Image
import torch
from torchvision import transforms
import threading
base_folder = os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".."))
if base_folder not in sys.path:
    sys.path.append(base_folder)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
augmentation_factor = 8
num_threads = 64
image_dir = os.path.abspath(os.path.join(base_folder, "./Data/train/image"))
label_dir = os.path.abspath(os.path.join(base_folder, "./Data/train/label"))
roi_dir = os.path.abspath(os.path.join(base_folder, "./Data/train/roi"))
output_dir = os.path.join(base_folder, "./Data/augmented_train")

# 图像增强转换器（与原始代码相同）
pre_transform = transforms.Compose([
    transforms.CenterCrop(1020),
    transforms.Pad(320, padding_mode="reflect"),
])
transform = transforms.Compose([
    transforms.RandomRotation(degrees=30, expand=True),
    transforms.CenterCrop(1200),
    transforms.RandomResizedCrop(
        1200, scale=(0.8, 1.0), ratio=(0.75, 1.333), antialias=True
    ),
    transforms.RandomCrop(1024),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
])
ToTensor = transforms.ToTensor()
ToPILImage = transforms.ToPILImage()

# 处理单个图像的函数


def process_image(filename, index, image_path, label_path):
    image_path = os.path.join(image_dir, filename)
    label_path = os.path.join(label_dir, filename)
    roi_path = os.path.join(roi_dir, filename)
    image = Image.open(image_path)
    label = Image.open(label_path)
    roi = Image.open(roi_path)
    image_tensor = ToTensor(image)
    label_tensor = ToTensor(label)
    roi_tensor = ToTensor(roi)
    image.close()
    label.close()
    roi.close()
    concatenated_tensor = pre_transform(
        torch.cat((image_tensor, label_tensor, roi_tensor), dim=0).to(device)
    )
    del image_tensor
    del label_tensor
    del roi_tensor
    for i in range(augmentation_factor):
        augmented_tensor = transform(concatenated_tensor).to("cpu")
        augmented_image = augmented_tensor[:3]
        augmented_label = augmented_tensor[3:4]
        augmented_roi = augmented_tensor[4:]
        del augmented_tensor
        image_output_path = os.path.join(
            output_dir, f"./image/{index * augmentation_factor + i + 1}.png")
        augmented_image = ToPILImage(augmented_image)
        augmented_image.save(image_output_path)
        del augmented_image
        augmented_label = (augmented_label > 0.5).to(torch.uint8) * 255
        label_output_path = os.path.join(
            output_dir, f"./label/{index * augmentation_factor + i + 1}.png")
        augmented_label = ToPILImage(augmented_label)
        augmented_label.save(label_output_path)
        del augmented_label
        augmented_roi = (augmented_roi > 0.5).to(torch.uint8) * 255
        roi_output_path = os.path.join(
            output_dir, f"./roi/{index * augmentation_factor + i + 1}.png")
        augmented_roi = ToPILImage(augmented_roi)
        augmented_roi.save(roi_output_path)
        del augmented_roi
        print(f"{index * augmentation_factor + i + 1}.png")


# 使用多进程处理图像
if __name__ == '__main__':
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, './image'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, './label'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, './roi'), exist_ok=True)

    filenames = [f for f in os.listdir(
        image_dir) if f.endswith(".jpg") or f.endswith(".png")]
    threads = []
    for i in range(len(filenames)):
        threads.append(threading.Thread(target=process_image,
                       args=(filenames[i], i, image_dir, label_dir)))
        threads[-1].start()
        if i >= num_threads:
            threads[i-num_threads].join()
