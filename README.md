# Plant Disease Detection

This project focuses on using deep learning to detect plant diseases, enhancing agricultural diagnostics. Leveraging CNN-based architectures like MobileNetV2 and EfficientNet, the system identifies plant diseases from images with high accuracy.

---

## Features
- **Deep Learning Model**: Employs MobileNetV2 and EfficientNet for robust and efficient disease classification.
- **Custom Dataset**: Built a Kaggle dataset with 22 distinct plant disease categories.
- **Performance Metrics**: Optimized using precision, recall, and F1-score.
- **Frameworks and Tools**: Python, OpenCV, PyTorch, and NumPy.

---

## Installation

### Requirements
Ensure you have the following dependencies installed:
- Python 3.7+
- OpenCV
- PyTorch
- NumPy

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/yogeshwaran245/plant-disease-detection.git
   cd plant-disease-detection
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the trained model weights and place them in the models directory.

---

## Usage

### Training the Model
To train the model on the custom dataset:
```bash
python train.py --dataset_path /path/to/dataset --epochs 50 --batch_size 32
```

### Testing the Model
To evaluate the model:
```bash
python test.py --model_path models/efficientnet_best.pth --test_data /path/to/test_data
```

### Running Inference
For predicting plant diseases from new images:
```bash
python inference.py --image_path /path/to/image.jpg --model_path models/efficientnet_best.pth
```

---

## Dataset
The dataset contains 22 plant disease categories, labeled and structured for training, validation, and testing. It was prepared using Kaggle, including augmentation techniques to improve model generalization.

### Example Categories
- Healthy
- Leaf Blight
- Bacterial Spot
- Powdery Mildew

---

## Model Architecture
### MobileNetV2
- Lightweight architecture optimized for mobile and embedded devices.

### EfficientNet
- Scalable and efficient CNN architecture that balances accuracy and computational cost.

---

## Results
- Achieved high accuracy across all 22 categories.
- Reliable performance metrics:
  - **Precision**: 92%
  - **Recall**: 91%
  - **F1-Score**: 91%

---

## Acknowledgments
- [PyTorch](https://pytorch.org/) for the deep learning framework.
- [Kaggle](https://kaggle.com/) for hosting the dataset.
- Agricultural diagnostic research for inspiration and motivation.
