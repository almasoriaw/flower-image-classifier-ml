# 🌸 Flower Image Classification

![License](https://img.shields.io/badge/license-MIT-blue)
![Python](https://img.shields.io/badge/python-3.6%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-1.7%2B-orange)

An advanced flower image classification system using pre-trained Convolutional Neural Networks (CNNs) for accurate species identification.

## 📋 Project Overview

This project implements a robust flower image classifier leveraging transfer learning with pre-trained CNN models. The system can identify 102 different flower species with high accuracy, making it useful for botanists, gardeners, and nature enthusiasts.

## 🌟 Features

- **Transfer Learning**: Utilizes pre-trained CNN architectures (VGG, ResNet, AlexNet)
- **High Accuracy**: Achieves excellent classification results on flower images
- **Command-line Interface**: Easy-to-use CLI for both training and prediction
- **Custom Configuration**: Adjust model hyperparameters via command-line arguments
- **GPU Support**: Option to use GPU acceleration for faster training and inference
- **Visualization**: Graphical display of prediction results with probabilities

## 🔍 Technical Overview

The application implements transfer learning with pre-trained CNN models from PyTorch's model zoo. These models have been trained on the ImageNet dataset with millions of images across thousands of categories.

Three CNN architectures are supported:
- **VGG**: Deep CNN known for its simplicity and effectiveness
- **ResNet**: Residual Network with skip connections for deeper network training
- **AlexNet**: Pioneering CNN architecture that won the 2012 ImageNet challenge

## 📁 Project Structure

```
flower-classifier/
├── data/                        # Data directory
│   └── flowers/                 # Flower image dataset
├── src/                         # Source code
│   ├── model_configuration.py   # Model setup and configuration
│   └── data_processing.py       # Data loading and preprocessing
├── utils/                       # Utility functions
│   └── get_input_args.py        # Command-line argument handling
├── images/                      # Project images for documentation
├── tests/                       # Test code
├── train.py                     # Training script
├── predict.py                   # Prediction script
└── requirements.txt             # Project dependencies
```

## 🛠️ Installation

1. Clone the repository:
   ```
   git clone https://github.com/almasoriaw/flower-image-classifier-ml.git
   cd flower-classifier
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. (Optional) Download the flower dataset:
   ```
   python train.py --data_directory flowers
   ```

## 🚀 Usage

### Training

```bash
python train.py --arch vgg13 --learning_rate 0.001 --hidden_units 512 --epochs 10 --training_compute gpu
```

Parameters:
- `--arch`: Model architecture (vgg11, vgg13, alexnet, resnet)
- `--learning_rate`: Learning rate for training
- `--hidden_units`: Number of hidden units
- `--epochs`: Number of training epochs
- `--training_compute`: Use 'cpu' or 'gpu' for training

### Prediction

```bash
python predict.py --checkpoint checkpoints/model_checkpoint --image_path path/to/image.jpg --top_k 5
```

Parameters:
- `--checkpoint`: Path to the saved model checkpoint
- `--image_path`: Path to the image for prediction
- `--top_k`: Number of top predictions to display

## 📊 Results

The model achieves accuracy of over 90% on the validation set with the VGG13 architecture after 10 epochs of training.

## 👩‍💻 Author

- **Alma Soria**
- **Date**: July 2024
- **LinkedIn**: [Connect with me](https://www.linkedin.com/in/almasoria/)
- **Portfolio**: [View my portfolio](https://github.com/almasoriaw/)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [PyTorch](https://pytorch.org/) for the excellent deep learning framework
- [102 Flower Dataset](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) by Visual Geometry Group at Oxford
