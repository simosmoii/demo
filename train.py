import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score

# ==========================================
# 1. 配置参数设置
# ==========================================
DATA_DIR = './datasets'          # 数据集根目录
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VALID_DIR = os.path.join(DATA_DIR, 'valid') 

BATCH_SIZE = 32                  # 批次大小 (如果显存/内存不足，可以调小为 16 或 8)
EPOCHS = 10                      # 训练轮数
LEARNING_RATE = 0.001            # 学习率
SAVE_PATH = './datasets/best_mobilenetv3.pth' # 最优模型保存路径

if __name__ == '__main__':
    # 检测是否有可用的 GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # ==========================================
    # 2. 数据预处理与加载
    # ==========================================
    # 训练集需要做数据增强以提高模型泛化能力
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),      # 随机裁剪并缩放
        transforms.RandomHorizontalFlip(),      # 随机水平翻转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # 随机颜色变化
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet 标准均值和方差
    ])

    # 验证集只需要中心裁剪和标准化，不需要增加随机性
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(VALID_DIR, transform=valid_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    num_classes = len(train_dataset.classes)
    print(f"检测到 {num_classes} 个鸟类类别。")
    print(f"训练集图片数量: {len(train_dataset)} | 验证集图片数量: {len(valid_dataset)}")

    # ==========================================
    # 3. 构建模型
    # ==========================================
    # 加载预训练的 MobileNetV3 (使用预训练权重可以大大加快收敛速度并提高精度)
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)

    # 修改最后一层全连接层以适应我们的鸟类类别数量
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)

    model = model.to(device)

    # ==========================================
    # 4. 定义损失函数和优化器
    # ==========================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # ==========================================
    # 5. 训练循环
    # ==========================================
    best_acc = 0.0
    history_train_loss = []
    history_valid_loss = []
    history_train_acc = []
    history_valid_acc = []

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train() # 切换到训练模式
        running_loss = 0.0
        train_corrects = 0
        
        train_bar = tqdm(train_loader, desc=f"第 {epoch+1}/{EPOCHS} 轮 [训练]")
        for inputs, labels in train_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()    # 清空梯度
            outputs = model(inputs)  # 前向传播
            loss = criterion(outputs, labels) # 计算损失
            loss.backward()          # 反向传播
            optimizer.step()         # 更新权重
            
            running_loss += loss.item() * inputs.size(0)
            
            _, preds = torch.max(outputs, 1)
            train_corrects += torch.sum(preds == labels.data)
            
            # 进度条后缀显示当前批次的 loss
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            
        epoch_loss = running_loss / len(train_dataset)
        train_acc = train_corrects.double() / len(train_dataset)
        
        # 验证阶段
        model.eval()
        corrects = 0
        valid_loss = 0.0
        
        with torch.no_grad(): # 验证时不计算梯度
            all_probs = []   # 用于收集这一轮的预测概率 (用于绘制PR曲线)
            all_labels = []  # 用于收集真实标签
            valid_bar = tqdm(valid_loader, desc=f"第 {epoch+1}/{EPOCHS} 轮 [验证]")
            for inputs, labels in valid_bar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                
                # 收集概率值和真实值，用于生成 PR 曲线
                probs = torch.nn.functional.softmax(outputs, dim=1)
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        epoch_acc = corrects.double() / len(valid_dataset)
        val_epoch_loss = valid_loss / len(valid_dataset)
        
        # 记录当前的 loss 和 accuracy 以备画图
        history_train_loss.append(epoch_loss)
        history_valid_loss.append(val_epoch_loss)
        history_train_acc.append(train_acc.item())
        history_valid_acc.append(epoch_acc.item())
        
        print(f"第 {epoch+1}/{EPOCHS} 轮 | 训练损失: {epoch_loss:.4f} | 验证损失: {val_epoch_loss:.4f} | 训练准确率: {train_acc:.4f} | 验证准确率: {epoch_acc:.4f}", flush=True)
        
        # 保存当前准确率最高的模型
        if epoch_acc > best_acc:
            best_acc = epoch_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f" -> 发现新的最优模型，已保存至 {SAVE_PATH}！", flush=True)

    print(f"训练完成！最高验证集准确率: {best_acc:.4f}")
    print(f"最优模型权重文件已保存在: {SAVE_PATH}")

    # ==========================================
    # 6. 生成并保存图表
    # ==========================================
    print("正在生成训练曲线和混淆矩阵图表...")
    
    # 1. 绘制 Loss 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, EPOCHS+1), history_train_loss, marker='o', label='Train Loss')
    plt.plot(range(1, EPOCHS+1), history_valid_loss, marker='o', color='red', label='Valid Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.savefig('./datasets/loss_curve.png')
    plt.close()

    # 2. 绘制 Accuracy 曲线
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, EPOCHS+1), history_train_acc, marker='o', color='blue', label='Train Acc')
    plt.plot(range(1, EPOCHS+1), history_valid_acc, marker='o', color='orange', label='Valid Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig('./datasets/accuracy_curve.png')
    plt.close()

    # 3. 绘制 PR 曲线 (Micro-average, 使用最后一轮的数据)
    Y_valid = label_binarize(all_labels, classes=range(num_classes))
    y_score = np.array(all_probs)
    
    precision, recall, _ = precision_recall_curve(Y_valid.ravel(), y_score.ravel())
    average_precision = average_precision_score(Y_valid, y_score, average="micro")
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='purple', label=f'Micro-average PR curve (AP={average_precision:.4f})')
    plt.title('Precision-Recall Curve (Micro-average)')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend(loc="lower left")
    plt.savefig('./datasets/pr_curve.png')
    plt.close()
    
    print("图表已成功保存至 ./datasets/ 目录下！")