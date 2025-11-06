import torch.cuda
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd.grad_mode import set_grad_enabled
import torch.nn as nn
import torch.optim as optim
import time
import copy
import warnings
from sklearn.metrics import roc_auc_score, confusion_matrix
from tqdm import tqdm
import argparse
from data import MyDataset
from util import *
np.seterr(divide='ignore')


# 定义训练和验证函数
def train_model(model, dataloaders, criterion, optimizer, save_dir, num_epochs=25):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'ckpt'), exist_ok=True)
    ckpt_dir = os.path.join(save_dir, 'ckpt')

    since = time.time()

    metrics = {'train_loss': [], 'valid_loss': [], 
               'train_acc': [], 'valid_acc': [],
               'train_auc': [], 'valid_auc': [],
               'train_sensitivity': [], 'valid_sensitivity': [],
               'train_specificity': [], 'valid_specificity': [],
               'train_ppv': [], 'valid_ppv': [],
               'train_npv': [], 'valid_npv': []
               }

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_auc = 0.0
    best_sensitivity = 0.0
    best_specificity = 0.0
    best_ppv = 0.0
    best_npv = 0.0
    best_epoch = 0

    for epoch in range(num_epochs):
        # 每个 epoch 包含训练和验证两个阶段
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train() 
            else:
                model.eval() 

            all_preds = []
            all_labels = []
            all_preds_prob = []
            running_loss = 0.0

            # 迭代数据
            for inputs, labels, _ in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = [int(label) for label in labels]
                labels = torch.tensor(labels).to(device)

                if phase == 'train': 
                    optimizer.zero_grad()  # 清零梯度

                # 前向传播
                with set_grad_enabled(phase == 'train'): # 只有在训练阶段才记录历史      
                    outputs = model(inputs)

                    if not isinstance(model, models.Inception3) and outputs.ndim == 4:  # SwinTransformer 输出的3维张量需要处理 [batch_size, width, height, num_classes] --> [batch_size, num_classes]
                        pooling_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
                        output_pooled = pooling_layer(outputs.permute(0, 3, 1, 2))
                        outputs = output_pooled.view(output_pooled.size(0), -1)

                    if isinstance(model, models.Inception3) and phase == 'train': # 对于 InceptionV3，InceptionOutputs不一样，需要单独处理
                        loss1 = criterion(outputs.logits, labels)
                        loss2 = criterion(outputs.aux_logits, labels)
                        outputs = outputs.logits
                        loss = loss1 + 0.4 * loss2  # aux_logits通常权重较小
                    else:
                        loss = criterion(outputs, labels)

                    if phase == 'train':                    
                        loss.backward()
                        optimizer.step()

                    probabilities = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(probabilities, dim=1)

                # 统计
                running_loss += loss.item() * inputs.size(0)

                all_labels.extend(labels.data.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_preds_prob.extend(probabilities[:, 1].detach().cpu().numpy())

            # print(all_preds_prob)
            # print(all_preds)

            auc_score = roc_auc_score(all_labels, all_preds_prob) # 输入真实label，预测为正例的概率
            cm = confusion_matrix(all_labels, all_preds)
            TN, FP, FN, TP = cm.ravel() 

            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                epoch_acc = (TP + TN) / (TP + TN + FP + FN)
                sensitivity = TP / (TP + FN)
                specificity = TN / (TN + FP)
                ppv = TP / (TP + FP)
                npv = TN / (TN + FN)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)

            # 记录指标
            if phase == 'train':
                metrics['train_loss'].append(epoch_loss)
                metrics['train_acc'].append(epoch_acc)
                metrics["train_auc"].append(auc_score)
                metrics['train_sensitivity'].append(sensitivity)
                metrics['train_specificity'].append(specificity)
                metrics['train_ppv'].append(ppv)
                metrics['train_npv'].append(npv)
            else:
                metrics['valid_loss'].append(epoch_loss)
                metrics['valid_acc'].append(epoch_acc)
                metrics["valid_auc"].append(auc_score)
                metrics['valid_sensitivity'].append(sensitivity)
                metrics['valid_specificity'].append(specificity)
                metrics['valid_ppv'].append(ppv)
                metrics['valid_npv'].append(npv)

            print(
                f"Epoch {epoch + 1}/{num_epochs} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {auc_score:.4f} Sensitivity: {sensitivity:.4f} Specificity: {specificity:.4f} PPV: {ppv:.4f} NPV: {npv:.4f}"
            )

            with open(os.path.join(save_dir, 'log.txt'), 'a') as f:
                f.write(
                    f"Epoch {epoch + 1}/{num_epochs} {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AUC: {auc_score:.4f} Sensitivity: {sensitivity:.4f} Specificity: {specificity:.4f} PPV: {ppv:.4f} NPV: {npv:.4f}\n"
                )

            # 深拷贝模型
            if phase == 'valid' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_auc = auc_score
                best_sensitivity = sensitivity
                best_specificity = specificity
                best_ppv = ppv
                best_npv = npv
                best_model_weight = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(ckpt_dir, "best_model.pth"))

        # 保存模型
        if epoch == num_epochs - 1:
            torch.save(model.state_dict(), os.path.join(ckpt_dir, 'last_model.pth'))

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f} AUC: {best_auc:4f} Sensitivity: {best_sensitivity:4f} Specificity: {best_specificity:4f} PPV: {best_ppv:4f} NPV: {best_npv:4f} at epoch {best_epoch}')

    with open(os.path.join(save_dir, 'train.log'), 'w') as f:
        for i in range(num_epochs):
            f.write(f'Epoch {i + 1}/{epoch} ')
            for key in metrics.keys():
                f.write(f' {key}: {metrics[key][i]} ')
            f.write('\n')

        f.write(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        f.write(f'Best epoch: {best_epoch} \n')
        f.write(f'Best val Acc: {best_acc:4f} AUC: {best_auc:.4f} Sensitivity: {best_sensitivity:.4f} Specificity: {best_specificity:.4f} PPV: {best_ppv:.4f} NPV: {best_npv:.4f}\n')

    # 画图
    plot_metrics(metrics, save_path, best_epoch)

    # 加载最佳模型权重
    model.load_state_dict(best_model_weight) # type: ignore

    return model


def test_model(model, dataloader, save_path):
    model.eval()  # 设置模型为评估模式

    running_corrects = 0
    all_preds = []
    all_preds_prob = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels, _ in dataloader:
            inputs = inputs.to(device)
            labels = [int(label) for label in labels]
            labels = torch.tensor(labels).to(device)

            outputs = model(inputs)

            if outputs.ndim == 4:  # SwinTransformer 输出的3维张量需要处理 [batch_size, width, height, num_classes] --> [batch_size, num_classes]
                pooling_layer = torch.nn.AdaptiveAvgPool2d((1, 1))
                output_pooled = pooling_layer(outputs.permute(0, 3, 1, 2))
                outputs = output_pooled.view(output_pooled.size(0), -1)

            probabilities = F.softmax(outputs, dim=1)
            preds = torch.argmax(probabilities, dim=1)

            running_corrects += torch.sum(preds == labels.data)
            all_labels.extend(labels.data.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_preds_prob.extend(probabilities[:, 1].detach().cpu().numpy())

    auc_score = roc_auc_score(all_labels, all_preds_prob)
    cm = confusion_matrix(all_labels, all_preds)
    TN, FP, FN, TP = cm.ravel()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        epoch_acc = (TP + TN) / (TP + TN + FP + FN)
        sensitivity = TP / (TP + FN)
        specificity = TN / (TN + FP)
        ppv = TP / (TP + FP)
        npv = TN / (TN + FN)

    import seaborn as sns
    from sklearn.metrics import roc_curve, auc

    # 绘制混淆矩阵
    plt.figure()
    plt.clf()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title("confusion_matrix")
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))

    # 计算ROC曲线的各个点
    fpr, tpr, thresholds = roc_curve(all_labels, all_preds_prob)
    roc_auc = auc(fpr, tpr)

    # 绘制AUC曲线
    plt.figure()
    plt.clf()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('TPR')
    plt.ylabel('FPR')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_path, 'roc_curve.png'))

    accuracy = running_corrects / len(dataloader.dataset)
    print(
        f"Test Accuracy: {epoch_acc:.4f} AUC: {auc_score:.4f}  Sensitivity: {sensitivity:.4f}  Specificity: {specificity:.4f}  PPV: {ppv:.4f}  NPV: {npv:.4f}"
    )
    with open(os.path.join(save_path, 'test.log'), 'w') as f:
        f.write(
            f"Test Accuracy: {accuracy:.4f} AUC: {auc_score:.4f}  Sensitivity: {sensitivity:.4f}  Specificity: {specificity:.4f}  PPV: {ppv:.4f}  NPV: {npv:.4f}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the classification model")
    parser.add_argument('--data', type=str, required=True, help='path to the directory')
    parser.add_argument('--save', type=str, required=True, help='path to save the model and plots')
    parser.add_argument('--model', type=str, required=True, help='name of the model. Choose from: AlexNet EfficientNet InceptionV3 ResNet50 VGG16 SwinTransformer ViT')
    parser.add_argument('--modality', type=str, required=True, help='Choose from: all contrast ultrasound')
    parser.add_argument('--area', type=str, required=True, help='Choose from: roi1 target')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--epoch', type=int, default=300, help='')
    parser.add_argument('--seed', type=int, default=36, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='')
    parser.add_argument('--device', type=str, default='1', help='')
    args = parser.parse_args()

    model_name = args.model # AlexNet InceptionV3 ResNet50 VGG16
    # data_path = os.path.join(args.data, args.area)
    
    save_path = os.path.join(args.save, f"{args.modality}-{args.lr}", model_name, args.area)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.device == '1':
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    elif args.device == '0':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = args.batch_size
    num_classes = 2  
    num_epochs = args.epoch
    lr = args.lr

    train_data_path = os.path.join(args.data, 'qinzhou1', args.modality, args.area)
    valid_data_path = os.path.join(args.data, 'qinzhou2', args.modality, args.area)
    test_data_path = os.path.join(args.data, 'hunan', args.modality, args.area)

    print(f'Parameters: \nmodel_name: {model_name}\ndata_path: {args.data}\nsave_path: {save_path}\nnum_epochs: {num_epochs}\nlr: {lr}')

    img_size = (224, 224)
    if args.model in ['InceptionV3']:
        img_size = (299, 299)
        
    train_loader, valid_loader, test_loader, model = None, None, None, None
    
    if args.modality == 'all':
        train_data_path1 = os.path.join(args.data, 'qinzhou1', 'ultrasound', args.area)
        valid_data_path1 = os.path.join(args.data, 'qinzhou2', 'ultrasound', args.area)
        test_data_path1 = os.path.join(args.data, 'hunan', 'ultrasound', args.area)
        
        train_data_path2 = os.path.join(args.data, 'qinzhou1', 'contrast', args.area)
        valid_data_path2 = os.path.join(args.data, 'qinzhou2', 'contrast', args.area)
        test_data_path2 = os.path.join(args.data, 'hunan', 'contrast', args.area)
        
        train_loader, _, _ = get_dataloader_multi_modal(train_data_path1, train_data_path2, img_size=img_size, split_ratio=[1, 0, 0], batch_size=args.seed)
        _ , valid_loader, _ = get_dataloader_multi_modal(valid_data_path1, valid_data_path2, img_size=img_size, split_ratio=[0, 1, 0], batch_size=args.seed)
        _, _, test_loader = get_dataloader_multi_modal(test_data_path1, test_data_path2, img_size=img_size, split_ratio=[0, 0, 1], batch_size=args.seed)
        
        model = get_model_channel(model_name, num_classes, device, 6)
    else:
        train_loader, _, _ = get_dataloader(train_data_path, img_size=img_size, split_ratio=[1, 0, 0], batch_size=args.seed)
        _, valid_loader, _ = get_dataloader(valid_data_path, img_size=img_size, split_ratio=[0, 1, 0], batch_size=args.seed)
        _, _, test_loader = get_dataloader(test_data_path, img_size=img_size, split_ratio=[0, 0, 1], batch_size=args.seed)
        model = get_model(model_name, num_classes, device, pretrained=False)
    
    print()
    print("TRAIN dataset:", valid_data_path)
    print("VALID dataset:", valid_data_path)
    print("TEST dataset:", test_data_path)
    print()
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.1*lr)
    dataloaders = {'train': train_loader, 'valid': valid_loader}

    best_model = train_model(model, dataloaders, criterion, optimizer, save_path, num_epochs)
    test_model(best_model, test_loader, save_path)
