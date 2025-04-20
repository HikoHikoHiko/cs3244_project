# CS3244 Machine Learning Project: Human Activity Recognition

## Overview

This project explores various Machine Learning architectures and preprocessing methods to classify human activities using data collected from smartphone inertial sensors. The dataset used is the **Smartphone-Based Recognition of Human Activities and Postural Transitions**, which includes 12 labeled activity types such as walking, standing, sitting, and transitions between these postures.

We evaluate the effect of different preprocessing strategies and compare the final accuracy of all models to identify the best model that suitable for prediction of the dataset.

## File Structure


###  Models
- **SVM**: Support Vector Machine classifier.
- **KNN**: K-Nearest Neighbors classifier.
- **MLP**: Multilayer Perceptron classifier.
- **XGBoost**: Extreme Gradient Boosting classifier.
- **RNN**: Recurrent Neural Network model.
- **BaseCNN**: Covolution Neuron Networks Model with raw dataset.
- **CNN_smotepca**: CNN model with smote and pca used.
- **CNN_smote**: CNN model with only SMOTE used.
- **SMOTE**: Processed original dataset with SMOTE.

## Dataset

- **Source**: [UCI HAR Dataset - Extended](http://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions)
- **Features**: 561 time- and frequency-domain features extracted from accelerometer and gyroscope signals.
- **Activities**:
  1. WALKING  
  2. WALKING_UPSTAIRS  
  3. WALKING_DOWNSTAIRS  
  4. SITTING  
  5. STANDING  
  6. LAYING  
  7. STAND_TO_SIT  
  8. SIT_TO_STAND  
  9. SIT_TO_LIE  
  10. LIE_TO_SIT  
  11. STAND_TO_LIE  
  12. LIE_TO_STAND  

Refer to `features_info.txt` and `activity_labels.txt` in the DataSet folder for further detail.

## User Guide

Make sure you have installed the following libraries to exercute all the code successfully:

  1. matplotlib
  2. Numpy
  3. pandas
  4. sklearn
  5. seaborn
  6. tensorflow
  7. imblearn
  8. math
  9. collections
  10. random 
  11. torch
  12. warnings

