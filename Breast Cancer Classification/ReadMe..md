# **Breast Cancer Classification: Logistic Regression vs. Linear Regression**

## **Overview**

This project explores the use of both **Logistic Regression** and **Linear Regression** for classifying tumours as benign or malignant using the breast cancer Wisconsin dataset. While linear regression is traditionally used for regression tasks, this project compares its performance with logistic regression, a classification-specific algorithm, to provide insights into their roles in predictive modelling.

## **Objective**

The goal of this project is to use predictive modelling techniques to classify breast tumours based on features in the dataset and compare the performance and applicability of **Linear Regression** and **Logistic Regression**.

## **Techniques**

### **1\. Linear Regression**

Linear regression is a supervised learning algorithm used to predict continuous outcomes based on independent variables. Although it's generally not used for classification, this project demonstrates its ability to model relationships between the features of the dataset.

### **2\. Logistic Regression**

Logistic regression is a classification algorithm designed for binary outcomes. In this project, it is used to classify tumours as either malignant (positive class) or benign (negative class) based on various features, estimating the probability of malignancy.

## **Methodology**

1. **Data Preparation**: Loading and preparing the breast cancer Wisconsin dataset for analysis.  
2. **Data Exploration**: Investigating and understanding the structure and key characteristics of the dataset.  
3. **Data Preprocessing**: Cleaning and preprocessing the data, including handling missing values, normalising data, and splitting into training and test sets.  
4. **Data Visualization**: Visualising relationships between different features and the target variable to better understand the dataset.  
5. **Model Building**: Building both linear regression and logistic regression models.  
6. **Model Evaluation**: Evaluating the models using accuracy, precision, recall, and other performance metrics.  
7. **Results and Conclusion**: Summarising the performance of both models and comparing their strengths and weaknesses in this context.

## **Dataset**

The dataset used is the **Breast Cancer Wisconsin Dataset**, which contains features extracted from breast tumours to predict whether a tumour is benign or malignant. The dataset includes the following features:

* Mean of the cell radius, texture, perimeter, area, smoothness, etc.  
* Compactness, concavity, symmetry, and fractal dimension  
* Diagnosis (target variable: benign or malignant)

You can access the dataset here.

## **Project Structure**

* **`data_exploration.ipynb`**: Code for exploring and visualising the dataset.  
* **`data_preprocessing.ipynb`**: Code for preparing and cleaning the dataset.  
* **`model_building.ipynb`**: Code for building both linear and logistic regression models.  
* **`model_evaluation.ipynb`**: Code for evaluating the performance of the models.

## **Requirements**

* Python 3.x  
* Pandas  
* NumPy  
* Scikit-learn  
* Matplotlib  
* Seaborn  
* Google Colab (for running the notebook)

## **Setup**

Clone this repository:  
bash  
Copy code  
`git clone https://github.com/yourusername/breast-cancer-classification.git`

1. Navigate to the project directory:  
   bash  
2. Copy code  
   `cd breast-cancer-classification`  
3. Open and run the Jupyter Notebooks in Google Colab for each phase of the project.

## **Results**

The results of the models include:

* **Linear Regression**: Analysing feature relationships but not suitable for classification.  
* **Logistic Regression**: Effective in classifying tumours with a focus on performance metrics such as accuracy, precision, recall, and F1-score.

## **Conclusion**

This project demonstrates the complementary roles of linear regression and logistic regression in analysing and classifying breast cancer tumours. Logistic regression is a more suitable algorithm for this task, while linear regression helps in understanding feature relationships.

## **License**

This project is licensed under the MIT License. See the LICENSE file for more details.

