# 🚢 Kaggle's Titanic Machine Learning Competition 🌊
This GitHub repository contains the lastest solution I made to the Kaggle Titanic problem, where the goal is to predict the survival of passengers aboard RMS Titanic.

## 📋 Problem Description:
The Titanic problem is a classic machine learning challenge that involves using data from some Titanic passengers to create a system that can predict if individuals survived or not.

## 📂 Repository Structure:
**data/**: This directory contains the dataset used for training and evaluation. The train.csv file includes the labeled data used for model training, while the test.csv file contains the unlabeled data used for predictions.  

**solutions/**: This directory contains the final solution script or notebook that combines all the steps and generates the predictions.  

**titanic_survival_NN.ipynb**: The notebook containing code that retrieves data from the .csv files, performs preprocessing, conducts exploratory data analysis (EDA), handles missing data, generates new features, utilizes a model for prediction, and presents the final results.

## 🤖 Solution:

The steps made were:
- Get data
- Exploratory Data Analysis & Preprocessing:  
    - Feature Engineering  
       - Normalizing features  
       - Creating Titles feature  
    - Discovering & Imputing null data  
        - Imputing Embarked (manually)  
        - Imputing Fare (with KNN)  
        - Imputing Age (with random forest regressor)
    - Feature Selection (Dropping useless columns)
    - Split data for training
- Use predictive model
    - Create the model
    - Train model
- Model evaluation

Currently, this model has a 0.78229 accuracy on the Kaggle platform dataset.

For a more complete documentation, take a look at the *titanic_survival_NN.ipynb* notebook [here](titanic_survival_NN.ipynb)!
