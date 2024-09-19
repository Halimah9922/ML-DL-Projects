# **Movie Poster Genre Classification with CNN**

## **Overview**

This project focuses on developing a Convolutional Neural Network (CNN) model to accurately classify movie posters by their genres. The goal is to create a robust multi-label classification system capable of predicting movie genres based on the visual content of posters. CNN is trained to learn features from movie posters and then use these features to assign appropriate genres to unseen posters.

## **Dataset**

The dataset consists of movie posters and their corresponding genres. Each movie can have multiple genres, making this a multi-label classification task. The dataset is split into training and testing subsets, ensuring that the model's performance is evaluated on unseen data.

### **Dataset Link**

You can access the dataset folder [here](https://drive.google.com/drive/folders/1F7nQJ1Dgd_Is2yTuae5lC-tpzPRvrBFo?usp=sharing)

### **Data Structure**

The dataset contains:

* Movie poster images  
* Genre labels (multi-label format)

The labels are converted into arrays to facilitate multi-label classification. Efficient input pipelines are constructed using TensorFlow's `tf.data` API, and data loading performance is enhanced using techniques like caching and prefetching.

## **Data Processing**

1. **Loading the Dataset**: The movie posters and their genres are loaded from a CSV file, which contains file paths and corresponding labels.  
2. **Data Splitting**: The dataset is split into training and testing subsets.  
3. **Label Transformation**: Genre labels are converted into one-hot encoded arrays to handle multi-label classification.

## **Model Definition**

A CNN model is built using the Keras Functional API. The architecture includes:

* Convolutional Layers  
* Max-Pooling Layers  
* Dropout Layers (to prevent overfitting)  
* Dense Layers

The model is compiled with the following configuration:

* **Optimizer**: Adam  
* **Loss Function**: Binary Cross-Entropy (since it's a multi-label classification)  
* **Metrics**: Precision and Recall

## **Model Training**

The model is trained for **40 epochs** on the training dataset, with validation on the test set. Two callbacks are implemented:

* **ModelCheckpoint**: Saves the best model weights based on validation loss.  
* **LearningRateScheduler**: Dynamically adjusts the learning rate during training to optimise model performance.

## **Model Evaluation**

After training, the model is evaluated on the test set using various metrics and visualisations:

* **Loss Curves**: Training and validation loss curves to assess convergence.  
* **Precision & Recall**: Performance is analysed by plotting precision and recall across epochs.  
* **Confusion Matrix**: To identify classification errors and confusion between genres.  
* **Class Balance Analysis**: A genre distribution analysis to understand class imbalance and its effect on performance.  
* **Genre-Specific Metrics**: Precision, recall, and F1-scores are computed for individual genres to evaluate performance on each class.

## **Results**

* **Precision**: 0.579 (On average, the model is 57.9% accurate when predicting a genre for a movie poster).  
* **Recall**: 0.201 (The model is able to correctly identify 20.1% of all instances for each genre).

### **Insights:**

* The model performs well in classifying certain genres but struggles with others, especially those that are underrepresented in the dataset (e.g., genres like Documentary or Musical).  
* Overrepresented genres like **Drama** may lead to a class imbalance that affects the model’s ability to generalise well for less frequent genres.

## **Conclusion**

The CNN model demonstrates good performance in movie genre classification based on posters, but improvements can be made. Future enhancements may include:

* Refining the model architecture or hyperparameters.  
* Applying more advanced data augmentation techniques to mitigate class imbalance.  
* Experimenting with more sophisticated models, such as transfer learning.

Overall, this project provides a solid foundation for genre classification using deep learning techniques and offers avenues for further research and improvement.

## **How to Run the Project**

1. **Set up the Dataset**: Download the dataset from the link above and organize it in a folder.  
2. **Run the Notebook**: Use Google Colab or your local environment to open the provided Jupyter Notebook.  
3. **Training and Evaluation**: Follow the instructions in the notebook to train and evaluate the model.

## **Dependencies**

* Python 3.x  
* TensorFlow 2.x  
* Keras  
* Matplotlib  
* NumPy  
* Pandas

To install the required libraries, run:

bash  
Copy code  
`pip install -r requirements.txt`

## **References**

* TensorFlow Documentation  
* Keras Functional API

