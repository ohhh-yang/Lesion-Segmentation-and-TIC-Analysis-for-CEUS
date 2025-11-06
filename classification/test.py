import warnings
import torch.cuda
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.autograd.grad_mode import set_grad_enabled
from sklearn.metrics import roc_auc_score, confusion_matrix
import torch.nn as nn
import torch.optim as optim
import time
import copy
import pandas as pd
import shutil
from tqdm import tqdm
import argparse
from data import MyDataset
from util import *
np.seterr(divide='ignore')



def test_model(model, dataloader, save_path):
    model.eval()  # 设置模型为评估模式
    running_corrects = 0
    all_preds = []
    all_preds_prob = []
    all_labels = []
    all_scores = []

    with torch.no_grad():
        for inputs, labels, file_names in dataloader:
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
            
            for i, file_name in enumerate(file_names):
                map = {}
                map['file_name'] = file_name
                map['true_label'] = labels[i].item()
                map['pred_label'] = preds[i].item()
                map['prob'] = probabilities[i][1].item()
                if map['true_label'] != map['pred_label']:
                    map['file_name'] = '_ERROR_' + map['file_name']
                all_scores.append(map)

    df = pd.DataFrame(all_scores)
    df.to_csv(os.path.join(save_path, 'preds.csv'), index=False)
    
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
    print(f'误判为恶性(1): {FP}  误判为良性(0): {FN}')
    with open(os.path.join(save_path, 'test.log'), 'w') as f:
        f.write(
            f"Test Accuracy: {accuracy:.4f} AUC: {auc_score:.4f}  Sensitivity: {sensitivity:.4f}  Specificity: {specificity:.4f}  PPV: {ppv:.4f}  NPV: {npv:.4f}"
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run the classification model")
    parser.add_argument('--path', type=str, required=True, help='all exp path')
    parser.add_argument('--data', type=str, required=True, help='test data folder')
    parser.add_argument('--batch_size', type=int, default=64, help='')
    parser.add_argument('--device', type=str, default='1', help='')
    parser.add_argument('--seed', type=int, default=36, help='')
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.device == '1':
        device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    elif args.device == '0':
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
    exps = os.listdir(args.path)
    
    batch_size = args.batch_size
    num_classes = 2 
    img_size = (224, 224)

    # 所有实验
    for exp in exps:
        modality = None
        if 'all' in exp.lower():
            modality = 'all'
        elif 'contrast' in exp.lower():
            modality = 'contrast'
        elif 'ultrasound' in exp.lower():
            modality = 'ultrasound'
        else:
            print('Modality not found: ', exp)
            exit()
            
        # 实验文件夹下面有模型实验
        model_names = os.listdir(os.path.join(args.path, exp))   
        for model_name in model_names:
            if model_name in ['InceptionV3']:
                img_size = (299, 299) 
            else:
                img_size = (224, 224)
            for area in ['roi1', 'target']:
                ckpt_path = os.path.join(args.path, exp, model_name, area, 'ckpt/best_model.pth')
                test_path = os.path.join(args.path, exp, model_name, area, 'test')

                # 多中心测试
                for cohort in ['qinzhou1', 'qinzhou2', 'hunan', 'shihezi']:
                    save_path = os.path.join(test_path, cohort)
                    
                    # 已经测试过，跳过
                    if os.path.exists(save_path) and os.listdir(save_path) is not []:
                        continue
                    
                    os.makedirs(save_path, exist_ok=True)
                    
                    model = None
                    if modality == 'all':  # 六通道数据集
                        test_data_path1 = os.path.join(args.data, cohort, 'ultrasound', area)
                        test_data_path2 = os.path.join(args.data, cohort, 'contrast', area)
                        print(img_size)
                        _, _, test_loader = get_dataloader_multi_modal(test_data_path1, test_data_path2, img_size=img_size, split_ratio=[0, 0, 1], batch_size=args.seed)
                        model = get_model_channel(model_name, num_classes, device, 6)
                    else:
                        test_data_path = os.path.join(args.data, cohort, modality, area)
                        _, _, test_loader = get_dataloader(test_data_path, img_size=img_size, split_ratio=[0, 0, 1], batch_size=args.seed)
                        model = get_model(model_name, num_classes, device, pretrained=False)
                      

                    print(f'Testing on {cohort}, save to {save_path}')  
                    
                    model.load_state_dict(torch.load(ckpt_path))
                    test_model(model, test_loader, save_path)
                    
                    print('\n')
                    # exit()
