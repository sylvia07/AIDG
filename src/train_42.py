# # 增加了计算效率统计功能（训练时间、GPU显存占用、推理速度FPS）
import torch
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import accuracy_score
import os
from model import MultiModalCLIPModel
import csv
from dataloader import get_dataloader
import torch.multiprocessing as mp
import random
import numpy as np
import logging
import time
from datetime import timedelta


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, seed=42,device='cuda'):
    best_accuracy = 0.0
    best_model_wts = None
    tmp_epoch = 0

    # ========== 新增效率统计变量 ==========
    total_train_time = 0.0
    peak_memory = 0
    batch_times = []
    batch_sizes = []
    # ====================================

    training_start = time.time()  # 训练开始时间

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        epoch_start = time.time()

        # 重置显存统计
        torch.cuda.reset_peak_memory_stats()

        for batch in train_loader:
            # ========== 记录batch开始时间 ==========
            batch_start = time.time()
            # ====================================

            images = batch['image'].to(device)
            text_inputs = batch['text_input'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()

            # 前向传播
            outputs = model(images, text_inputs)
            loss = criterion(outputs, labels)

            # 反向传播
            loss.backward()
            optimizer.step()

            # ========== 记录batch效率 ==========
            torch.cuda.synchronize()  # 确保CUDA操作完成
            batch_time = time.time() - batch_start
            batch_times.append(batch_time)
            batch_sizes.append(len(images))
            # ====================================

            # 更新训练统计
            running_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total_samples += labels.size(0)

            # 更新峰值显存
            current_memory = torch.cuda.max_memory_allocated()
            if current_memory > peak_memory:
                peak_memory = current_memory

        # 更新学习率
        scheduler.step()

        # 计算epoch时间
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time

        # 计算指标
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples

        # val_accuracy = validate(model, val_loader, device)

        # if val_accuracy > best_accuracy:
        #     best_accuracy = val_accuracy
        #     best_model_wts = model.state_dict()
        #     tmp_epoch = epoch

        # ========== 计算训练效率指标 ==========
        avg_batch_latency = np.mean(batch_times[-len(train_loader):]) * 1000  # 毫秒
        avg_fps = np.sum(batch_sizes[-len(train_loader):]) / np.sum(batch_times[-len(train_loader):])
        # ====================================

        # 记录日志（新增效率指标）
        logging.info(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Acc: {epoch_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
            f"Epoch Time: {timedelta(seconds=int(epoch_time))} | "
            f"Batch Latency: {avg_batch_latency:.1f}ms | "
            f"Train FPS: {avg_fps:.1f}"
        )
        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Loss: {epoch_loss:.4f} | "
            f"Acc: {epoch_accuracy:.4f} | "
            f"Val Acc: {val_accuracy:.4f} | "
        )
    # ========== 最终效率报告 ==========
    total_train_time = time.time() - training_start
    peak_memory_gb = peak_memory / (1024 ** 3)

    efficiency_report = (
        f"\n=== Training Efficiency Report ===\n"
        f"Total Training Time: {timedelta(seconds=int(total_train_time))}\n"
        f"Peak GPU Memory Usage: {peak_memory_gb:.2f}GB\n"
        f"Average Batch Latency: {np.mean(batch_times) * 1000:.1f}ms\n"
        f"Average Train FPS: {np.sum(batch_sizes) / np.sum(batch_times):.1f}\n"
        f"\n=== Best Model ===\n"
        f"Best Val Accuracy: {best_accuracy:.4f} at Epoch {tmp_epoch + 1}"
    )

    print(efficiency_report)
    logging.info(efficiency_report)
    # ====================================

    model.load_state_dict(best_model_wts)
    torch.save(best_model_wts, f"best_model_{seed}.pth")
    return model


# def validate(model, val_loader, device='cuda'):
#     model.eval()
#     all_preds = []
#     all_labels = []

#     with torch.no_grad():
#         for batch in val_loader:
#             images = batch['image'].to(device)
#             text_inputs = batch['text_input'].to(device)
#             labels = batch['label'].to(device)

#             outputs = model(images, text_inputs)
#             preds = torch.argmax(outputs, dim=1)
#             all_preds.extend(preds.cpu().numpy())
#             all_labels.extend(labels.cpu().numpy())

#     # 保存预测结果
#     with open('train_val_predictions.csv', mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['True Label', 'Predicted Label'])
#         for true_label, pred_label in zip(all_labels, all_preds):
#             writer.writerow([true_label, pred_label])

#     return accuracy_score(all_labels, all_preds)


# if __name__ == "__main__":
#     mp.set_start_method("spawn", force=True)
#     seed = 123
#     set_seed(seed)
#     logging.info(f"Starting training with seed: {seed}")
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     logging.info(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
#
#     # 记录初始显存
#     torch.cuda.reset_peak_memory_stats()
#     initial_memory = torch.cuda.memory_allocated()
#     logging.info(f"Initial GPU Memory: {initial_memory / (1024 ** 3):.2f}GB")
#
#     # 设置数据路径和加载器
#     project_root = os.path.dirname(os.path.abspath(__file__))
#     project_root = os.path.abspath(os.path.join(project_root, '..'))
#
#     data_dir = 'data'
#     image_dir_train = os.path.join(project_root, data_dir, 'images/train/')
#     caption_dir_train = os.path.join(project_root, data_dir, 'captions/train/')
#     label_to_class = os.path.join(project_root, data_dir, 'label_to_class.json')
#     image_dir_test = os.path.join(project_root, data_dir, 'images/test/')
#     caption_dir_test = os.path.join(project_root, data_dir, 'captions/test/')
#
#     train_loader = get_dataloader(image_dir_train, caption_dir_train, label_to_class, batch_size=16)
#     val_loader = get_dataloader(image_dir_test, caption_dir_test, label_to_class, batch_size=16)
#     # train_loader = get_dataloader(image_dir_train, caption_dir_train, label_to_class, batch_size=8)
#     # val_loader = get_dataloader(image_dir_test, caption_dir_test, label_to_class, batch_size=8)
#
#     # 初始化模型和优化器
#     model = MultiModalCLIPModel(num_classes=28, device=device).to(device)
#     optimizer = optim.AdamW(model.parameters(), lr=1.1e-3, weight_decay=1e-3) # batch_size=16  dropout=0.4  Accuracy: 0.7161 Precision: 0.7532 Recall: 0.7161 F1 Score: 0.7223
#     scheduler = StepLR(optimizer, step_size=5, gamma=0.5)
#     criterion = torch.nn.CrossEntropyLoss()
#
#     # 训练模型
#     trained_model = train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=50,
#                           device=device)
#     logging.info("Training completed.")
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)

    # ======== 固定基本超参数 ========
    # seed = 77
    # seed = 9999
    # seed = 0
    # seed = 999
    seed = 42
    batch_size = 16
    lr = 1.1e-3
    weight_decay = 1e-3
    num_epochs = 50
    step_size = 5
    gamma = 0.5

    # 配置日志记录
    logging.basicConfig(
        filename=f'training_log_{seed}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # ======== 设置随机种子 ========
    print("Using seed:", seed)
    set_seed(seed)
    logging.info(f"Starting training with seed: {seed}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

    # ======== 路径与数据加载 ========
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(project_root, '..'))

    data_dir = 'data'
    image_dir_train = os.path.join(project_root, data_dir, 'images/train/')
    caption_dir_train = os.path.join(project_root, data_dir, 'captions/train/')
    label_to_class = os.path.join(project_root, data_dir, 'label_to_class.json')
    # image_dir_test = os.path.join(project_root, data_dir, 'images/test/')
    # caption_dir_test = os.path.join(project_root, data_dir, 'captions/test/')

    train_loader = get_dataloader(image_dir_train, caption_dir_train, label_to_class, batch_size=batch_size)
    # val_loader = get_dataloader(image_dir_test, caption_dir_test, label_to_class, batch_size=batch_size)
    # val_loader = {}
    # ======== 初始化模型和优化器 ========
    model = MultiModalCLIPModel(num_classes=28, device=device).to(device)

    # 自动提取 Dropout 值（默认取第一个 nn.Dropout 层）
    dropout_value = None
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            dropout_value = module.p
            break
    if dropout_value is None:
        dropout_value = "N/A"  # fallback

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    criterion = torch.nn.CrossEntropyLoss()

    # ======== 超参数记录（科学计数法） ========
    hyperparams = {
        "seed": seed,
        "batch_size": batch_size,
        "learning_rate": f"{lr:.2e}",
        "weight_decay": f"{weight_decay:.2e}",
        "num_epochs": num_epochs,
        "dropout": dropout_value,
        "step_size": step_size,
        "lr_decay_gamma": gamma,
    }

    logging.info("========== Hyperparameter Configuration ==========")
    for key, value in hyperparams.items():
        logging.info(f"{key}: {value}")
    logging.info("===================================================")

    # ======== 显存记录与训练开始 ========
    torch.cuda.reset_peak_memory_stats()
    initial_memory = torch.cuda.memory_allocated()
    logging.info(f"Initial GPU Memory: {initial_memory / (1024 ** 3):.2f}GB")

    # ======== 训练模型 ========
    trained_model = train(model, train_loader, val_loader, criterion, optimizer, scheduler,
                          num_epochs=num_epochs, seed=seed, device=device)
    logging.info("Training completed.")
