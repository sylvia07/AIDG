# # 添加时间效率等信息统计
import torch
import os
from model import MultiModalCLIPModel
from dataloader import get_dataloader
import torch.multiprocessing as mp
import csv
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import logging
import random
import time
from datetime import timedelta
import seaborn as sns
import matplotlib.pyplot as plt


# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# 绘制混淆矩阵的函数，并保存为文件
def plot_confusion_matrix(y_true, y_pred, class_names, title='Confusion Matrix', cmap='Blues',
                          save_path='confusion_matrix.png'):
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)

    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()

    # 保存图像
    plt.savefig(save_path, dpi=300)
    plt.show()  # 显示图像


# 测试模型
def test(model, test_loader, device='cuda'):
    model.eval()
    all_preds = []
    all_labels = []

    # ========== 新增效率统计变量 ==========
    total_inference_time = 0.0
    total_samples = 0
    peak_memory = 0
    # ====================================

    # 预热运行（避免首次运行的开销影响计时）
    with torch.no_grad():
        warmup_batch = next(iter(test_loader))
        _ = model(warmup_batch['image'].to(device), warmup_batch['text_input'].to(device))

    # 重置显存统计
    torch.cuda.reset_peak_memory_stats()
    test_start_time = time.time()

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            text_inputs = batch['text_input'].to(device)
            labels = batch['label'].to(device)

            # ========== 记录批处理开始时间 ==========
            batch_start = time.time()
            # ====================================

            # 前向传播
            outputs = model(images, text_inputs)

            # ========== 同步CUDA操作并记录时间 ==========
            torch.cuda.synchronize()
            batch_time = time.time() - batch_start
            total_inference_time += batch_time
            total_samples += len(images)
            # ====================================

            # 更新峰值显存
            current_memory = torch.cuda.max_memory_allocated()
            if current_memory > peak_memory:
                peak_memory = current_memory

            # 获取预测结果
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # ========== 计算效率指标 ==========
    total_test_time = time.time() - test_start_time
    avg_latency = (total_inference_time * 1000) / total_samples  # 毫秒
    fps = total_samples / total_inference_time
    peak_memory_gb = peak_memory / (1024 ** 3)
    # ====================================

    # 计算评估指标
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=1)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=1)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=1)

    # ========== 构建完整报告 ==========
    efficiency_report = (
        f"\n=== Test Efficiency Report ===\n"
        f"Total Test Time: {timedelta(seconds=int(total_test_time))}\n"
        f"Peak GPU Memory Usage: {peak_memory_gb:.2f}GB\n"
        f"Average Latency per Sample: {avg_latency:.1f}ms\n"
        f"Inference FPS: {fps:.1f}\n"
        f"\n=== Performance Metrics ===\n"
        f"Test Accuracy: {accuracy:.4f}\n"
        f"Test Precision: {precision:.4f}\n"
        f"Test Recall: {recall:.4f}\n"
        f"Test F1 Score: {f1:.4f}"
    )
    # ====================================

    # 打印并记录报告
    print(efficiency_report)
    logging.info(efficiency_report)

    # 保存混淆矩阵
    save_path = f'confusion_matrix_seed_{seed}.png'  # 每次使用不同的文件名以避免覆盖
    plot_confusion_matrix(all_labels, all_preds, class_names=[f"Class {i}" for i in range(28)], save_path=save_path)

    # 保存预测结果
    with open('test_predictions.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['True Label', 'Predicted Label'])
        for true_label, pred_label in zip(all_labels, all_preds):
            writer.writerow([true_label, pred_label])

    return accuracy, precision, recall, f1


if __name__ == "__main__":

    # 设置随机种子
    seed = 42
    set_seed(seed)
    # 配置日志记录
    logging.basicConfig(
        filename=f'test_log_{seed}.log',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logging.info(f"Starting testing with seed: {seed}")
    mp.set_start_method("spawn", force=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 记录设备信息
    if torch.cuda.is_available():
        logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logging.info(f"CUDA Version: {torch.version.cuda}")
        # 记录初始显存
        torch.cuda.reset_peak_memory_stats()
        initial_memory = torch.cuda.memory_allocated()
        logging.info(f"Initial GPU Memory Usage: {initial_memory / (1024 ** 3):.2f}GB")
    else:
        logging.info("Using CPU")

    # 设置数据路径
    project_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(project_root, '..'))

    data_dir = 'data'
    image_dir_test = os.path.join(project_root, data_dir, 'images/test/')
    caption_dir_test = os.path.join(project_root, data_dir, 'captions/test/')
    label_to_class = os.path.join(project_root, data_dir, 'label_to_class.json')

    # 创建数据加载器
    test_loader = get_dataloader(image_dir_test, caption_dir_test, label_to_class, batch_size=16)

    # 初始化模型
    model = MultiModalCLIPModel(num_classes=28, device=device).to(device)

    # 加载训练好的模型权重
    model.load_state_dict(torch.load(f'best_model_{seed}.pth'))
    # model.load_state_dict(torch.load('best_model_77.pth'))

    # 测试模型
    accuracy, precision, recall, f1 = test(model, test_loader, device=device)
    logging.info("Testing completed.")

