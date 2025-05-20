
# Cotton Leaf Disease Classification using VGG16, VGG19, and ResNet50

This project performs image classification on a cotton leaf disease dataset using transfer learning with pre-trained models (VGG16, VGG19, and ResNet50). It uses TensorFlow and Keras for building and training the models, and visualizes the predictions on test images.

##  Dataset

- **Dataset**: [Cotton Leaf Disease Dataset](https://www.kaggle.com/datasets/seroshkarim/cotton-leaf-disease-dataset)
- The dataset contains images of healthy and diseased cotton leaves categorized into classes.

##  Features

- Preprocessing and automatic dataset splitting into training and testing sets.
- Transfer learning with:
  - VGG16
  - VGG19
  - ResNet50
- Visualization of predictions.
- Data augmentation using `ImageDataGenerator`.

##  Requirements

Install the following Python packages:

```bash
pip install tensorflow numpy matplotlib scikit-learn kaggle
Also, make sure you have your Kaggle API key in kaggle.json for dataset download.

 How to Run
Upload your kaggle.json to the working directory.

Run the script:

bash
Copy
Edit
python vgg.py
This will:

Download the dataset from Kaggle.

Unzip and split the dataset into training and test sets.

Train models using VGG16, VGG19, and ResNet50.

Show prediction visualizations.

 Models Used
VGG16: 16-layer CNN with ImageNet weights

VGG19: 19-layer CNN with ImageNet weights

ResNet50: 50-layer deep residual network

All models use the following setup:

Input size: (224, 224, 3)

Classification layer with softmax activation

Loss: categorical_crossentropy

Optimizer: adam

Evaluation: Accuracy on test data

 Output
The script will display:

Training and validation accuracy and loss.

Image predictions with actual and predicted labels.
