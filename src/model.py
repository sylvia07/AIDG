import os
import torch
import torch.nn as nn
import clip
from PIL import Image
from torch.cuda.amp import autocast

# 修复后的多模态 CLIP 模型
class MultiModalCLIPModel(nn.Module):
    def __init__(self, num_classes, device, image_feature_size=768, text_feature_size=768):
    # def __init__(self, num_classes, device, image_feature_size=512, text_feature_size=512):
        super(MultiModalCLIPModel, self).__init__()

        # 使用CLIP加载模型和tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clip_model_path = "/data2/wll/baseline_plantdoc_l14_pvd/src/clip/ViT-L-14.pt"  # 本地路径

        # 确保模型从本地路径加载
        if os.path.exists(clip_model_path):
            self.clip_model, self.preprocess = clip.load(clip_model_path, self.device)
        else:
            raise FileNotFoundError(f"CLIP 模型文件未找到: {clip_model_path}")

        # CLIP模型中的图像和文本特征提取部分
        self.image_encoder = self.clip_model.encode_image
        self.text_encoder = self.clip_model.encode_text

        # 投影层：将CLIP图像和文本特征投影到统一的维度
        self.visual_proj = nn.Linear(image_feature_size, 512)  # 将视觉特征投影到较小维度
        self.text_proj = nn.Linear(text_feature_size, 512)  # 将文本特征投影到较小维度

        # PVD部分
        self.pvd_fc1 = nn.Linear(1024, 256)  # 输入维度调整为 1024
        self.pvd_fc2 = nn.Linear(256, 128)  # PVD模块的第二个投影层
        self.pvd_fc3 = nn.Linear(128, 64)   # 最终输出层

        # 调整 PVD 输出到 512 维
        self.pvd_proj = nn.Linear(64, 512)

        # 分类网络
        self.fc1 = nn.Linear(512, 1024)  # 相加后的 512 维特征
        # self.dropout = nn.Dropout(0.2)
        # self.dropout = nn.Dropout(0.3) # 最好 Accuracy: 0.7119
        # self.dropout = nn.Dropout(0.4) #acc 0.7161
        self.dropout = nn.Dropout(0.4) # Accuracy: 0.7119
        # self.dropout = nn.Dropout(0.5) #Accuracy: 0.7119 Precision: 0.7470 Recall: 0.7119 F1 Score: 0.7157
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, image, text):
        # 确保 text 张量的形状正确：去除大小为 1 的第二维
        text = text.squeeze(1)

        with torch.no_grad():  # 禁止梯度计算，节省内存
            # 图像特征提取
            image_features = self.image_encoder(image)  # [batch_size, image_feature_size]
            # 文本特征提取
            text_features = self.text_encoder(text)  # [batch_size, text_feature_size]

        # 使用 autocast 进行混合精度训练
        with autocast():
            # 投影到相同维度
            visual_features = torch.relu(self.visual_proj(image_features))  # [batch_size, 512]
            text_features = torch.relu(self.text_proj(text_features))  # [batch_size, 512]

            # PVD模块
            pvd_features = torch.cat((visual_features, text_features), dim=1)  # [batch_size, 1024]
            pvd_out = torch.relu(self.pvd_fc1(pvd_features))  # [batch_size, 256]
            pvd_out = torch.relu(self.pvd_fc2(pvd_out))  # [batch_size, 128]
            pvd_out = torch.relu(self.pvd_fc3(pvd_out))  # [batch_size, 64]

            # 将 PVD 输出调整到 512 维
            pvd_out = torch.relu(self.pvd_proj(pvd_out))  # [batch_size, 512]

            # 确保数据类型一致
            visual_features = visual_features.float()
            text_features = text_features.float()
            pvd_out = pvd_out.float()

            # 特征相加
            combined_features = visual_features + text_features + pvd_out  # [batch_size, 512]

        # 分类部分
        x = torch.relu(self.fc1(combined_features))  # [batch_size, 1024]
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, num_classes]
        return x

