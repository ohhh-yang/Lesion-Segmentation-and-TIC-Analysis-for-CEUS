from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.model_selection import train_test_split
from torchvision.datasets import ImageFolder
from PIL import Image
import os
import torch
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# 定义数据集类，输入是路径列表，输出是图片、标签、文件名
class MyDataset(Dataset):
    def __init__(self, data_paths, transform=None,  img_size=(224, 224), mode: str = 'train'):
        self.data_path = data_paths

        if mode == 'train' and transform is None:
            self.transform = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(size, scale=(0.9, 1.0)),
                    transforms.Resize(img_size),
                    transforms.RandomRotation(30),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif (mode == 'valid' or mode == 'test')  and transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.data_path)

    def __getitem__(self, idx):
        image = Image.open(self.data_path[idx])
        img = self.transform(image)
        label = os.path.dirname(self.data_path[idx]).split('/')[-1]
        file_name = os.path.basename(self.data_path[idx])
        return img, label, file_name



class MultiModalDataset(Dataset):
    def __init__(self, img_paths1, img_paths2, transform=None, img_size=(224, 224), mode='train'):
        
        assert len(img_paths1) == len(img_paths2), "The two modalities must have the same images"
        img_names1 = [os.path.basename(f) for f in img_paths1]
        img_names2 = [os.path.basename(f) for f in img_paths2]
        assert set(img_names1) == set(img_names2), "The two modalities must have the same images"
        
        self.img_paths1 = img_paths1
        self.img_paths2 = img_paths2
        
        if mode == 'train' and transform is None:
            self.transform = transforms.Compose(
                [
                    # transforms.RandomResizedCrop(size, scale=(0.9, 1.0)),
                    transforms.Resize(img_size),
                    transforms.RandomRotation(30),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif (mode == 'valid' or mode == 'test')  and transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize(img_size),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif transform is not None:
            self.transform = transform

    def __len__(self):
        return len(self.img_paths1)

    def __getitem__(self, idx):
        img1_path = self.img_paths1[idx]
        img2_path = self.img_paths2[idx]
        image1 = Image.open(img1_path)
        image2 = Image.open(img2_path)

        # Ensure both images are in RGB format
        image1 = image1.convert('RGB')
        image2 = image2.convert('RGB')

        img_trans1 = self.transform(image1)
        img_trans2 = self.transform(image2)

        # Combine the two images into a 6-channel image
        combined_image = torch.cat((img_trans1, img_trans2), dim=0)

        # Get the class label from the parent directory name
        dir_name = os.path.basename(os.path.dirname(img1_path))
        class_label = os.path.basename(dir_name)
        img_name = os.path.basename(img1_path)

        return combined_image, class_label, img_name
