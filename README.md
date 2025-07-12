# Boosting Crop Disease Recognition via Automated Image Description Generation and Multimodal Fusion

This project implements a multimodal classification model that combines image and text information using the CLIP (Contrastive Language-Image Pre-training) framework and enhances feature learning through the PVD (Projected Visual-Textual Discriminant) module.

---

## Project Structure
```
├── data/ 
│   ├── images/                 
│   │   ├── test/ 
│   │   │  ├── Apple_leaf
│   │   │  │  ├── Apple_leaf_00001.jpg
│   │   │  │  ├── Apple_leaf_00002.jpg
│   │   │  ├── Apple_rust_leaf
│   │   │  │  ├── Apple_rust_leaf_00001.jpg
│   │   │  │  ├── Apple_rust_leaf_00002.jpg
│   │   ├── train/
│   │   │  ├── Apple_leaf
│   │   │  │  ├── Apple_leaf_00001.jpg
│   │   │  │  ├── Apple_leaf_00002.jpg
│   │   │  ├── Apple_rust_leaf
│   │   │  │  ├── Apple_rust_leaf_00001.jpg
│   │   │  │  ├── Apple_rust_leaf_00002.jpg
│   ├── captions/           
│   │   ├── test
│   │   │  ├── Apple_rust_leaf.json
│   │   ├── train
│   │   │  ├── Apple_rust_leaf.json
│   ├── label_to_class.json  
│   ├── src/ 
│   │   │── model.py
│   │   │── train.py
│   │   │── test.py
│   │   │── dataloader.py
├── dataset_PlantDoc/
├── captions_CogAgent/
├── captions_LLaVA/
├── description_CogAgent.py
├── description_LLaVA.py

Demo of captions:
{
    "Apple_leaf_00006.jpg": {
        "text": "the color of the leaf.",
        "label": 0
    },
    "Apple_leaf_00008.jpg": {
        "text": "The petiole of the leaf is thin and green, and it appears to be slightly curved.",
        "label": 0
    }}
```
---

## Core Components

### 1. **Multimodal CLIP Model (`MultiModalCLIPModel`)**

This model extracts image and text features using CLIP, projects them to a common space, fuses them via the PVD module, and performs classification through a fully connected network. Key parts include:

* **Image Feature Extractor**: Extracts features from images using the CLIP image encoder.
* **Text Feature Extractor**: Extracts textual embeddings using the CLIP text encoder.
* **PVD Module**: Projects both modalities to a shared space, fuses them, and enhances discriminability.
* **Classifier**: Predicts class labels based on the fused features.

#### ViT-L/14 Model Notice

Due to GitHub’s file size limitation (100MB), the file `clip/ViT-L-14.pt` is not included in this repository.

You can obtain the model from:

1. Hugging Face: [https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K)
2. Official OpenAI: Load via `clip.load("ViT-L/14")` and save manually.

---

### 2. **PVD (Projected Visual-Textual Discriminant) Module**

The PVD module applies further projections and non-linear transformations after feature fusion, enabling more discriminative learning and improving classification accuracy.
---

## Dataset

This project assumes the dataset is organized as follows:
You can obtain the dataset from: [https://github.com/pratikkayal/PlantDoc-Dataset](https://github.com/pratikkayal/PlantDoc-Dataset)
```
data/
├── images/
│   ├── train/
│   └── test/
├── captions/
│   ├── train/
│   └── test/
└── label_to_class.json
```

* `images/train/`: Training images
* `images/test/`: Test images
* `captions/train/`: Text descriptions corresponding to training images
* `captions/test/`: Text descriptions for test images
* `label_to_class.json`: Maps numeric labels to class names

---

## Dependencies

* `torch`: Deep learning framework
* `transformers`: For loading CLIP models
* `torchvision`: Image preprocessing utilities
* `scikit-learn`: Performance metrics
* `matplotlib`, `seaborn`: Visualization of confusion matrices and training results

---

## Installation

1. Install PyTorch and required packages:

```bash
Python 3.8.20

pip install torch torchvision transformers scikit-learn matplotlib seaborn
```

2. Download and configure the CLIP model:

   * Place the CLIP model file (e.g., `ViT-L-14.pt`) under the appropriate project directory (`src/clip/`).

---

## Training

Before training, ensure your dataset is placed as described. Then run:

```bash
python training.py
```


