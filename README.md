# Multi-Class Prediction of Cirrhosis Outcomes

## Project Overview

This project aims to predict the outcomes of patients with cirrhosis using a multi-class classification model. The model is trained on a data from the Kaggel Playground Series - Season 3, Episode 26. The trained model is deployed to an Azure endpoint for easy access and inference.

## Table of Contents

- [Project Overview](#project-overview)
- [Data](#data)
- [Model](#model)
- [Deployment](#deployment)
- [Usage](#usage)

## Data

Walter Reade, Ashley Chow. (2023). Multi-Class Prediction of Cirrhosis Outcomes. Kaggle. https://kaggle.com/competitions/playground-series-s3e26

## Model

We use a multi-class classification model to predict the following outcomes:
- `D` - Deceased
- `CL` - Alive due to liver a transplant
- `C` - Continue with the disease

The model is built using the following steps:
1. Data Preprocessing: Handling missing values, encoding categorical variables, and feature scaling.
2. Model Training: Using Ensamble fo XGBosst and LightGBM.

## Deployment

The trained model is deployed on Microsoft Azure using Azure Machine Learning services. The model endpoint can be accessed for real-time predictions. (Note: Endpoint offline to save costs)

### Azure Deployment Steps

2. **Register Model**: Register the trained model in the Azure ML workspace.
3. **Deploy Model**: Deploy the registered model to an Azure endpoint.
4. **Test Endpoint**: Ensure the endpoint is working correctly by sending test data and receiving predictions.

## Usage

To use the deployed model for predictions, send a POST request to the Azure endpoint with the patient's clinical data in JSON format.
