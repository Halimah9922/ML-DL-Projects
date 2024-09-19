# **Handwritten Digit Recognition with CNN**

## **Overview**

This repository contains a machine learning project focused on recognizing handwritten digits using the MNIST dataset. The project leverages a Convolutional Neural Network (CNN) to achieve high accuracy in digit classification. Implemented in Google Colab, this project demonstrates advanced machine learning techniques, model evaluation, and error analysis.

## **Dataset**

The MNIST dataset comprises 60,000 training images and 10,000 test images of handwritten digits (0-9). The dataset is sourced from Kaggle and is provided in four files:

* `train-images-idx3-ubyte`: Training images  
* `train-labels-idx1-ubyte`: Training labels  
* `t10k-images-idx3-ubyte`: Test images  
* `t10k-labels-idx1-ubyte`: Test labels

You can find the dataset [here](https://www.kaggle.com/datasets/hojjatk/mnist-dataset).

## **Setup and Execution**

1. **Upload the Dataset**: Upload the `mnist_dataset` folder, containing the MNIST dataset files, to your Google Drive.  
2. **Open Google Colab**: Open the provided Jupyter Notebook in Google Colab.  
3. **Run the Notebook**: Execute the code cells to preprocess the data, build and train the CNN model, and analyze the results.

## **Project Structure**

* **`data_preprocessing.ipynb`**: Contains code for preprocessing the MNIST dataset, including normalisation and reshaping.  
* **`model_building_training.ipynb`**: Defines and trains the CNN model using TensorFlow’s Keras API.  
* **`model_evaluation_analysis.ipynb`**: Evaluates the trained model, reports performance metrics, and conducts error analysis.

## **Results and Analysis**

* **Model Training**: Visualises the training and validation loss/accuracy over epochs.  
* **Model Evaluation**: Reports accuracy and provides a classification report with precision, recall, and F1-score.  
* **Error Analysis**: Includes visualisations of misclassified images and a confusion matrix to assess the model's performance.

## **Requirements**

* Python 3.x  
* TensorFlow 2.x  
* NumPy  
* Matplotlib  
* Google Colab (for running the notebook)

