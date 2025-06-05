# Cotton Leaf Disease Classification with VGG16, VGG19, and ResNet50

![Cotton Leaf Disease](https://via.placeholder.com/800x200.png?text=Cotton+Leaf+Disease+Classification)  
*Classifying cotton leaf diseases using deep learning models: VGG16, VGG19, and ResNet50*

## Project Overview

This project implements deep learning models—VGG16, VGG19, and ResNet50—to classify diseases in cotton leaves. Using the [Cotton Leaf Disease Dataset](https://www.kaggle.com/datasets/seroshkarim/cotton-leaf-disease-dataset) from Kaggle, these models identify various disease types (e.g., bacterial blight), aiding farmers in early detection and crop health management. The project compares the performance of these architectures to determine the most effective model for this task.

### Features
- **Multiple Models**: Implements VGG16, VGG19, and ResNet50 for robust image classification.
- **Kaggle Dataset Integration**: Seamlessly downloads and processes the cotton leaf disease dataset.
- **GPU Acceleration**: Optimized for training on GPU environments like Google Colab.
- **Sample Predictions**: Visualizes model predictions on test images.
- **Comprehensive Documentation**: Step-by-step guide for setup, training, and evaluation.

## Repository Structure

```
cotton-leaf-disease-classification/
├── data/
│   └── COTTON/                     # Extracted dataset (not tracked in Git)
├── notebooks/
│   └── VGG.ipynb                   # Main Jupyter notebook with model implementations
├── scripts/
│   └── preprocess_data.py          # Data preprocessing script
├── models/
│   ├── vgg16_model.h5             # Trained VGG16 model (not tracked in Git)
│   ├── vgg19_model.h5             # Trained VGG19 model (not tracked in Git)
│   └── resnet50_model.h5          # Trained ResNet50 model (not tracked in Git)
├── images/
│   ├── sample_prediction1.jpg      # Sample predicted image
│   └── sample_prediction2.jpg      # Sample predicted image
├── README.md                       # Project overview and instructions
├── requirements.txt                # Python dependencies
└── LICENSE                         # License file
```

## Installation

### Prerequisites
- Python 3.8+
- Google Colab with GPU (optional, for faster training)
- Kaggle API key (for dataset download)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/cotton-leaf-disease-classification.git
   cd cotton-leaf-disease-classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Up Kaggle API**:
   - Download your `kaggle.json` from Kaggle.
   - Place it in the `~/.kaggle/` directory and set permissions:
     ```bash
     mkdir -p ~/.kaggle
     cp kaggle.json ~/.kaggle/
     chmod 600 ~/.kaggle/kaggle.json
     ```

4. **Download Dataset**:
   ```bash
   kaggle datasets download -d seroshkarim/cotton-leaf-disease-dataset
   unzip cotton-leaf-disease-dataset.zip -d ./data/COTTON
   ```

## Usage

1. **Run the Jupyter Notebook**:
   Open `notebooks/VGG.ipynb` in Jupyter or Google Colab. The notebook includes:
   - Installation of required packages.
   - Dataset downloading and preprocessing.
   - Training and evaluation of VGG16, VGG19, and ResNet50 models.
   - Visualization of predictions.

2. **Run Preprocessing Script** (optional):
   ```bash
   python scripts/preprocess_data.py
   ```

3. **Model Training**:
   - Ensure GPU is enabled in Colab for faster training.
   - Adjust hyperparameters in `VGG.ipynb` for each model (VGG16, VGG19, ResNet50) as needed.

4. **Inference**:
   - Use the trained models (`models/vgg16_model.h5`, `models/vgg19_model.h5`, `models/resnet50_model.h5`) to classify new cotton leaf images.

## Dataset

The [Cotton Leaf Disease Dataset](https://www.kaggle.com/datasets/seroshkarim/cotton-leaf-disease-dataset) contains images of cotton leaves categorized by disease types, such as bacterial blight. The dataset is organized into subfolders by class, suitable for multi-class image classification.

## Results

- **Model Performance**:
  - **VGG16**: (Add accuracy, e.g., ~82% on validation set)
  - **VGG19**: (Add accuracy, e.g., ~84% on validation set)
  - **ResNet50**: (Add accuracy, e.g., ~86% on validation set)
- **Comparison**: (Briefly describe which model performed best after your experiments)

### Sample Predictions

Below are example predictions from the models on test images from the dataset:

| Image | True Label | Predicted Label (VGG16) | Predicted Label (VGG19) | Predicted Label (ResNet50) |
|-------|------------|-------------------------|-------------------------|----------------------------|
| ![Sample 1](images/sample_prediction1.jpg) | Bacterial Blight | Bacterial Blight | Bacterial Blight | Bacterial Blight |
| ![Sample 2](images/sample_prediction2.jpg) | Healthy | Healthy | Healthy | Bacterial Blight |

*Note*: Replace `sample_prediction1.jpg` and `sample_prediction2.jpg` with actual images from your model predictions. Upload these to the `images/` folder in the repository.

## Contributing

Contributions are welcome! Please:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/YourFeature`).
3. Commit your changes (`git commit -m 'Add YourFeature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For questions or feedback, reach out via:
- GitHub: [yourusername](https://github.com/yourusername)
- Email: your.email@example.com

---

*Happy coding, and let's help farmers keep their cotton crops healthy with advanced deep learning models!*