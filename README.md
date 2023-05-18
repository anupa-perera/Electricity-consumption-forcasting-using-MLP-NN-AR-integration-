*Electricity Consumption Forecasting using MLP-NN (AR Integration)


This project focuses on predicting the next day's electricity consumption for the University Building at 115 New Cavendish Street, London, based on hourly data. The dataset provided contains electricity consumption data in kWh for the years 2018 and 2019, specifically for three hours (20:00, 19:00, and 18:00). The goal is to develop a multilayer neural network (MLP-NN) model that utilizes the autoregressive (AR) approach to forecast the electricity consumption for the 20:00 hour case.

To get started, make sure you have the "UoW.xlsx" file, which includes the daily electricity consumption data. The dataset consists of a total of 470 samples.

The project will be divided into subtasks as follows:

Data Preparation:

Load the dataset from the "UoW_consumption.xlsx" file.
Extract the relevant attributes for the 20:00 hour case.
Split the data into a training set (first 380 samples) and a testing set (remaining samples).

Model Development:
Implement a multilayer neural network (MLP) model.
Use the autoregressive (AR) approach by considering time-delayed values of the 20th hour attribute as input variables.
Train the MLP model using the training set.


Model Evaluation:
Test the trained MLP model using the testing set.
Evaluate the performance of the model by comparing the predicted electricity consumption with the actual consumption.
Calculate appropriate performance metrics, such as mean squared error (MSE) or root mean squared error (RMSE), to assess the accuracy of the predictions.

Next Day Electricity Consumption Forecasting:
Utilize the trained MLP model to forecast the electricity consumption for the next day at 20:00 hours.
Provide the predicted value as the output.
By following these steps, we aim to develop an accurate MLP-NN model that can effectively predict the next day's electricity consumption for the specified hour. This forecasting capability can assist in optimizing energy management and planning for the University Building at 115 New Cavendish Street, London.

Note: For a more detailed explanation, please refer to the project documentation and code files available in the GitHub repository.
