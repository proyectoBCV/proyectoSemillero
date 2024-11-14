# DermAI: Automated Dermatological ​Image Classification using CNNs​
# Project Overview
Skin cancer, a critical health issue globally, manifests in various forms, including non-melanoma (less aggressive) and melanoma (highly aggressive) cases. Although melanoma accounts for only 1% of all skin cancer cases, it is responsible for over 80% of skin cancer-related deaths. Early and accurate diagnosis of melanoma remains challenging, largely due to its visual similarities with other skin conditions.

This project uses dermoscopic images from the ISIC 2019 Challenge dataset to train and evaluate deep learning models for the automatic detection and classification of skin cancer. The dataset includes a total of 25,331 images across nine diagnostic categories:

Melanoma (MEL)
Melanocytic nevus (NV)
Basal cell carcinoma (BCC)
Actinic keratosis (AK)
Benign keratosis (BKL)
Dermatofibroma (DF)
Vascular lesion (VASC)
Squamous cell carcinoma (SCC)
The dataset is divided into:

Training set: 17,732 images (70%)
Validation set: 3,800 images (15%)
Test set: 3,799 images (15%)
Each image is annotated with a numerical label (0-8) representing its diagnostic category. This project aims to address the difficulty in distinguishing melanoma from other lesions by developing and training deep learning models that can identify the most discriminative features for early-stage melanoma detection.

The ISIC 2019 challenge provides a valuable benchmark for classification performance in dermoscopic image analysis, making this project relevant for advancing automated tools in skin cancer diagnosis.


# Methodology
This project employs convolutional neural networks (CNNs) such as AlexNet, ResNet, and Vision Transformer (ViT) in PyTorch for the classification of dermoscopic images. The methodology includes the following steps:

# Data Preprocessing
Images are resized and normalized. Data augmentation techniques, such as rotation, flipping, and scaling, are applied to improve model generalization.
# Model Implementation:
- AlexNet and ResNet (pretrained models) are fine-tuned on the ISIC dataset to leverage learned features.
- Vision Transformer (ViT) is also evaluated for its effectiveness in image classification tasks involving complex skin lesion images.
- Training and Optimization: Models are trained on the processed dataset with early stopping and hyperparameter tuning to minimize overfitting.
- Evaluation: Each model's performance is evaluated on the validation and test sets using metrics like accuracy, F1 score, and AUC (Area Under the Curve).

# Installation
- Clone the repository:
git clone https://github.com/your-username/DermAI.git
cd DermAI

- Install dependencies:
pip install -r requirements.txt
Download ISIC 2019 dataset: The dataset can be downloaded from the ISIC Archive and should be organized in the following structure:


DermAI/
├── data/
│   ├── train/
│   ├── valid/
│   └── test/

# Usage
To train and evaluate models, you can use the provided scripts:

- Train a model:
python train_model.py --model <model_name> --epochs <num_epochs>

- Evaluate a model:
python evaluate_model.py --model <model_name> --checkpoint <path_to_checkpoint>
Replace <model_name> with AlexNet, ResNet, or ViT as required.

# References
ISIC 2019 Challenge: https://www.isic-archive.com/
Skin Cancer Statistics: Global Cancer Observatory (GCO), World Health Organization (WHO).
