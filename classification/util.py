import os
from sklearn.model_selection import train_test_split
import torch.cuda
from torch.utils.data import DataLoader
from torchvision import models
import torch.nn as nn
import torch
import timm
from sklearn.metrics import roc_auc_score, confusion_matrix
from matplotlib import pyplot as plt
from data import MyDataset, MultiModalDataset


def get_dataloader(
    data_path, split_ratio, img_size=(224, 224), batch_size=32, num_workers=4, seed=36
):
    # 划分数据集 train / val / test 路径列表
    file_paths = []
    class_names = os.listdir(data_path)
    class_paths = [os.path.join(data_path, name) for name in class_names]

    for path in class_paths:
        if os.path.isdir(path):   
            file_paths += [os.path.join(path, file) for file in os.listdir(path)]

    for file in file_paths:
        if not file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.bmp')):
            file_paths.remove(file)
            
    train_len = int(len(file_paths) * split_ratio[0])
    valid_len = int(len(file_paths) * split_ratio[1])
    test_len = len(file_paths) - train_len - valid_len
    len_list = [train_len, valid_len, test_len]
    index_list = ['train', 'valid', 'test']
    
    if 0 in len_list:
        cnt_0 = 0
        index = 0
        for l in len_list:
            if l == 0:
                cnt_0 += 1
            else:
                index = len_list.index(l)
        if cnt_0 == 2:
            dataset = MyDataset(file_paths, img_size=img_size, mode='test')
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            label_0 = 0
            label_1 = 0
            for i, (img, label, file_name) in enumerate(loader):
                for label in label:
                    if label == '0':
                        label_0 += 1
                    elif label == '1':
                        label_1 += 1
            print(f'{index_list[index]}: 0-{label_0} 1-{label_1}')
            if index == 0:
                return loader, None, None
            elif index == 1:
                return None, loader, None
            elif index == 2:
                return None, None, loader
            

    train_paths, valid_test = train_test_split(file_paths, test_size=split_ratio[1] + split_ratio[2], random_state=seed)
    valid_paths, test_paths = train_test_split(valid_test, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=seed)

    # 获取dataloader
    train_dataset = MyDataset(train_paths, img_size=img_size, mode='train')
    valid_dataset = MyDataset(valid_paths, img_size=img_size, mode='valid')
    test_dataset = MyDataset(test_paths, img_size=img_size, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    print('Paths: ', data_path)
    print('Dataset: ', len(train_dataset), len(valid_dataset), len(test_dataset))
    print('DataLoader: ', len(train_loader), len(valid_loader), len(test_loader))

    files = []

    label_0 = 0
    label_1 = 0
    for i, (img, label, file_name) in enumerate(train_loader):
        files.extend(file_name)
        for label in label:
            if label == '0':
                label_0 += 1
            elif label == '1':
                label_1 += 1
    print(f'Train: 0-{label_0} 1-{label_1}')
    print('Files: ', len(files), files)

    files.clear()
    label_0 = 0
    label_1 = 0
    for i, (img, label, file_name) in enumerate(valid_loader):
        files.extend(file_name)
        for label in label:
            if label == '0':
                label_0 += 1
            elif label == '1':
                label_1 += 1
    print(f'Valid: 0-{label_0} 1-{label_1}')
    print('Files: ', len(files), files)

    files.clear()
    label_0 = 0
    label_1 = 0
    for i, (img, label, file_name) in enumerate(test_loader):
        files.extend(file_name)
        for label in label:
            if label == '0':
                label_0 += 1
            elif label == '1':
                label_1 += 1
    print(f'Test: 0-{label_0} 1-{label_1}')
    print('Files: ', len(files), files)

    return train_loader, valid_loader, test_loader


def get_dataloader_multi_modal(
    modality1_dir, modality2_dir, split_ratio, img_size=(224, 224), batch_size=32, num_workers=4, seed=36
):    
    # 两个模态类别相同
    class_names1 = os.listdir(modality1_dir)
    class_names2 = os.listdir(modality2_dir)
    assert set(class_names1) == set(class_names2), 'Class names in two modalities should be same'
    
    
    class_paths1 = [os.path.join(modality1_dir, name) for name in class_names1]
    class_paths2 = [os.path.join(modality2_dir, name) for name in class_names2]
    class_paths1.sort()
    class_paths2.sort()

    # 两个模态每个类别下文件相同
    for path1, path2 in zip(class_paths1, class_paths2):
        if os.path.isdir(path1) and os.path.isdir(path2):
            file_names1 = os.listdir(path1)
            file_names2 = os.listdir(path2)
            assert set(file_names1) == set(file_names2), f'File names in two modalities should be same: {path1} {path2}'


    file_paths1 = []
    file_paths2 = []
    for path1, path2 in zip(class_paths1, class_paths2):
        if os.path.isdir(path1) and os.path.isdir(path2):
            file_paths1 += [os.path.join(path1, file) for file in os.listdir(path1)]
            file_paths2 += [os.path.join(path2, file) for file in os.listdir(path2)]

    for file1, file2 in zip(file_paths1, file_paths2):
        if not file1.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.bmp')):
            file_paths1.remove(file1)
            file_paths2.remove(file2)
            
            
    train_len = int(len(file_paths1) * split_ratio[0])
    valid_len = int(len(file_paths1) * split_ratio[1])
    test_len = len(file_paths1) - train_len - valid_len
    len_list = [train_len, valid_len, test_len]
    index_list = ['train', 'valid', 'test']
    
    if 0 in len_list:
        cnt_0 = 0
        index = 0
        for l in len_list:
            if l == 0:
                cnt_0 += 1
            else:
                index = len_list.index(l)
        if cnt_0 == 2:
            dataset = MultiModalDataset(file_paths1, file_paths2, img_size=img_size, mode='test')
            loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
            label_0 = 0
            label_1 = 0
            for i, (img, label, file_name) in enumerate(loader):
                for label in label:
                    if label == '0':
                        label_0 += 1
                    elif label == '1':
                        label_1 += 1
            print(f'{index_list[index]}: 0-{label_0} 1-{label_1}')
            if index == 0:
                return loader, None, None
            elif index == 1:
                return None, loader, None
            elif index == 2:
                return None, None, loader

    train_paths1, valid_test1 = train_test_split(file_paths1, test_size=split_ratio[1] + split_ratio[2], random_state=seed)
    valid_paths1, test_paths1 = train_test_split(valid_test1, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=seed)
    
    train_paths2, valid_test2 = train_test_split(file_paths2, test_size=split_ratio[1] + split_ratio[2], random_state=seed)
    valid_paths2, test_paths2 = train_test_split(valid_test2, test_size=split_ratio[2] / (split_ratio[1] + split_ratio[2]), random_state=seed)

    # 获取dataloader
    train_dataset = MultiModalDataset(train_paths1, train_paths2, img_size=img_size, mode='train')
    valid_dataset = MultiModalDataset(valid_paths1, valid_paths2, img_size=img_size, mode='valid')
    test_dataset = MultiModalDataset(test_paths1, test_paths2, img_size=img_size, mode='test')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    print('Paths: ', modality1_dir, modality2_dir)
    print('Dataset: ', len(train_dataset), len(valid_dataset), len(test_dataset))
    print('DataLoader: ', len(train_loader), len(valid_loader), len(test_loader))

    files = []

    label_0 = 0
    label_1 = 0
    for i, (img, label, file_name) in enumerate(train_loader):
        files.extend(file_name)
        for label in label:
            if label == '0':
                label_0 += 1
            elif label == '1':
                label_1 += 1
    print(f'Train: 0-{label_0} 1-{label_1}')
    print('Files: ', len(files), files)

    files.clear()
    label_0 = 0
    label_1 = 0
    for i, (img, label, file_name) in enumerate(valid_loader):
        files.extend(file_name)
        for label in label:
            if label == '0':
                label_0 += 1
            elif label == '1':
                label_1 += 1
    print(f'Valid: 0-{label_0} 1-{label_1}')
    print('Files: ', len(files), files)

    files.clear()
    label_0 = 0
    label_1 = 0
    for i, (img, label, file_name) in enumerate(test_loader):
        files.extend(file_name)
        for label in label:
            if label == '0':
                label_0 += 1
            elif label == '1':
                label_1 += 1
    print(f'Test: 0-{label_0} 1-{label_1}')
    print('Files: ', len(files), files)

    return train_loader, valid_loader, test_loader



def get_model(model_name, num_classes, device, pretrained=True):
    model = None

    # AlexNet
    if model_name == 'AlexNet':
        model = models.alexnet(pretrained=pretrained)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes) # type: ignore
        model = model.to(device)

    # InceptionV3
    elif model_name == 'InceptionV3':
        model = models.inception_v3(init_weights=True, pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

    # ResNet50
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

    # VGG16
    elif model_name == 'VGG16':
        model = models.vgg16(pretrained=pretrained)        
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes) # type: ignore
        model = model.to(device)

    # EfficientNet
    elif model_name == 'EfficientNet':
        model = models.efficientnet_b0(pretrained=pretrained) # type: ignore
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model = model.to(device)

    # SwinTransformer
    elif model_name == "SwinTransformer":
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_classes)  # 修改最后的分类层
        model = model.to(device)
        
    # ViT
    elif model_name == "ViT":
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        model.head = nn.Linear(model.head.in_features, num_classes)  # 修改最后的分类层
        model = model.to(device)

    else:
        raise ValueError('Invalid model name: {}'.format(model_name))

    return model


def get_model_channel(model_name, num_classes, device, in_channels=3):
    model = None
    
    if in_channels != 3:
        print("Changing the input channels of the model to", in_channels)
        pretrained = False
        print("Pretrained set to False")

    # AlexNet
    if model_name == 'AlexNet':
        model = models.alexnet(pretrained=pretrained)
        # 修改第一个卷积层的输入通道数
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        model = model.to(device)

    # InceptionV3
    elif model_name == 'InceptionV3':
        model = models.inception_v3(init_weights=True, pretrained=pretrained)
        # 修改第一个卷积层的输入通道数
        model.Conv2d_1a_3x3.conv = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

    # ResNet50
    elif model_name == 'ResNet50':
        model = models.resnet50(pretrained=pretrained)
        # 修改第一个卷积层的输入通道数
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

    # VGG16
    elif model_name == 'VGG16':
        model = models.vgg16(pretrained=pretrained)
        # 修改第一个卷积层的输入通道数
        model.features[0] = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
        model = model.to(device)

    # EfficientNet
    elif model_name == 'EfficientNet':
        model = models.efficientnet_b0(pretrained=pretrained)
        # 修改第一个卷积层的输入通道数
        model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        model = model.to(device)

    # SwinTransformer
    elif model_name == "SwinTransformer":
        model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained)
        # 修改第一个卷积层的输入通道数
        model.patch_embed.proj = nn.Conv2d(in_channels, model.patch_embed.proj.out_channels, kernel_size=model.patch_embed.proj.kernel_size, stride=model.patch_embed.proj.stride, padding=model.patch_embed.proj.padding)
        model.head = nn.Linear(model.head.in_features, num_classes)
        model = model.to(device)
    
    # ViT
    elif model_name == "ViT":
        model = timm.create_model('vit_base_patch16_224', pretrained=pretrained)
        # 修改第一个卷积层的输入通道数
        model.patch_embed.proj = nn.Conv2d(in_channels, model.patch_embed.proj.out_channels, kernel_size=model.patch_embed.proj.kernel_size, stride=model.patch_embed.proj.stride, padding=model.patch_embed.proj.padding)
        model.head = nn.Linear(model.head.in_features, num_classes)
        model = model.to(device)

    else:
        raise ValueError('Invalid model name: {}'.format(model_name))

    return model



def plot_metrics(metrics, save_dir, best_epoch):

    # 绘制损失和准确率曲线
    plt.figure(figsize=(20, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(metrics['train_loss'], label='Train Loss')
    plt.plot(metrics['valid_loss'], label='Val Loss')
    plt.axvline(best_epoch, color='r', linestyle='--', label='Best Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.plot(metrics['train_acc'], label='Train Acc')
    plt.plot(metrics['valid_acc'], label='Val Acc')
    plt.axvline(best_epoch, color='r', linestyle='--', label=f'Epoch{best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 3, 3)
    plt.plot(metrics['train_auc'], label='Train AUC')
    plt.plot(metrics['valid_auc'], label='Val AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()

    plt.subplot(2, 3, 4)
    plt.plot(metrics['train_sensitivity'], label='Train Sensitivity')
    plt.plot(metrics['valid_sensitivity'], label='Val Sensitivity')
    plt.xlabel('Epoch')
    plt.ylabel('Sensitivity')
    plt.legend()

    plt.subplot(2, 3, 5)
    plt.plot(metrics['train_specificity'], label='Train Specificity')
    plt.plot(metrics['valid_specificity'], label='Val Specificity')
    plt.xlabel('Epoch')
    plt.ylabel('Specificity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_acc.png'))
