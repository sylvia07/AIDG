import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import re
from torchvision import transforms

class MultimodalDataset(Dataset):
    def __init__(self, image_dir, caption_dir, label_to_class_file, transform=None, max_seq_len=50):
        """
        :param image_dir: 图像文件夹路径
        :param caption_dir: 文本描述文件夹路径
        :param label_to_class_file: 标签到类的映射文件
        :param transform: 图像变换
        :param max_seq_len: 最大文本长度（用于填充/截断）
        """
        self.image_dir = image_dir
        self.caption_dir = caption_dir
        self.label_to_class = self._load_label_to_class(label_to_class_file)
        self.transform = transform
        self.file_names = self._get_file_names()
        self.max_seq_len = max_seq_len

        # 使用CLIP加载模型和tokenizer
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        clip_model_path = "/data2/wll/baseline_plantdoc_l14_pvd/src/clip/ViT-L-14.pt"  # 本地路径

        # 确保模型从本地路径加载
        if os.path.exists(clip_model_path):
            self.clip_model, self.preprocess = clip.load(clip_model_path, self.device)
        else:
            raise FileNotFoundError(f"CLIP 模型文件未找到: {clip_model_path}")

        # 使用 CLIP 内置的 tokenizer
        self.tokenizer = clip.tokenize

    def _load_label_to_class(self, label_to_class_file):
        """加载标签到类的映射"""
        with open(label_to_class_file, 'r') as f:
            return json.load(f)

    def _get_file_names(self):
        """获取所有文件名"""
        folder_names = os.listdir(self.image_dir)
        file_names = []
        for folder in folder_names:
            folder_path = os.path.join(self.image_dir, folder)
            if os.path.isdir(folder_path):
                for file_name in os.listdir(folder_path):
                    if file_name.endswith('.jpg'):
                        file_names.append((folder, file_name))
        return file_names

    def _load_caption(self, folder, image_name):
        """加载每张图片的文本描述"""
        image_caption_name = self.rename_file(image_name)
        caption_file = os.path.join(self.caption_dir, f'{image_caption_name}.json')
        if not os.path.exists(caption_file):
            raise FileNotFoundError(f"未找到对应的文本描述文件: {caption_file}")
        with open(caption_file, 'r') as f:
            caption_data = json.load(f)
        if image_name not in caption_data:
            raise KeyError(f"描述文件中未找到对应的图片: {image_name}")
        return caption_data[image_name]

    def rename_file(self, filename):
        """去除文件名中的数字和扩展名"""
        return re.sub(r'_\d+\.jpg$', '', filename)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        folder, image_name = self.file_names[idx]
        image_path = os.path.join(self.image_dir, folder, image_name)

        # 加载图像
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"未找到图像文件: {image_path}")
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # 加载文本描述和标签
        caption = self._load_caption(folder, image_name)
        text = caption['text']
        label = caption['label']

        # 图像预处理
        image_input = self.preprocess(image).to(self.device)

        # 文本预处理
        text_input = self.tokenizer([text], truncate=True).to(self.device)

        return {
            'image': image_input,
            'text_input': text_input,
            'label': torch.tensor(label).long()
        }


def get_dataloader(image_dir, caption_dir, label_to_class_file, batch_size=32):
    dataset = MultimodalDataset(image_dir, caption_dir, label_to_class_file)
    print(dataset)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.RandomResizedCrop(224),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),# 色彩抖动
    transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),  # 加入仿射变换
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])