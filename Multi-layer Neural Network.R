#load dataset
library(readxl)
data <- read_excel("C:/Users/Anupa/Desktop/ml cw/uow.xlsx")

#-------------------------------------Data cleaning Phase----------------------#

#Check the structure of the dataset
str(data)

#Check for missing values
summary(data) #no N/A's

#Check for duplicate rows in the dataset
duplicate_rows <- duplicated(data)

#Count the number of duplicate rows
num_duplicate_rows <- sum(duplicate_rows)

#Print the number of duplicate rows
cat("Number of duplicate rows:", num_duplicate_rows)

#------------------------------------------------------------------------------#

#--------------------Data transformation phase---------------------------------#

#convert date column to a date format
data$date <- as.Date(data$date, format="%m/%d/%Y")

#convert 20:00 column data  to a time series object
library(xts)
data_xts <- xts(data$`20:00`, order.by=as.Date(data$date, format="%m/%d/%Y"))

#Create time-delayed input vectors up to (t-7)
input_t1 <- data.frame(t_1 = lag(data_xts, 1))
input_t2 <- data.frame(t_2 = lag(data_xts, 2))
input_t3 <- data.frame(t_3 = lag(data_xts, 3))
input_t4 <- data.frame(t_4 = lag(data_xts, 4))
input_t5 <- data.frame(t_5 = lag(data_xts, 5))
input_t6 <- data.frame(t_6 = lag(data_xts, 6))
input_t7 <- data.frame(t_7 = lag(data_xts, 7))

# Create output vector
output <- data.frame(output = data_xts)

#====================================================#
#combine I/O data frames to form I/O matrix for (t-1)
io_matrix1 <- cbind(input_t1, output)

#Remove rows with NA values
io_matrix1 <- na.omit(io_matrix1)

#normalize I/O matrix of(t-1) using z-score
io_matrix_norm1 = scale(io_matrix1)
#====================================================#

#combine I/O data frames to form I/O matrix for (t-2)
io_matrix2 <- cbind(input_t1, input_t2, output)

# Remove rows with NA values
io_matrix2 <- na.omit(io_matrix2)


#normalize I/O matrix of(t-2) using z-score
io_matrix_norm2 = scale(io_matrix2)

#=====================================================#
#combine I/O data frames to form I/O matrix for (t-3)
io_matrix3 <- cbind(input_t1, input_t2, input_t3, output)

#Remove rows with NA values
io_matrix3 <- na.omit(io_matrix3)

#normalize I/O matrix of(t-3) using z-score
io_matrix_norm3 <- scale(io_matrix3)


#======================================================#

#combine I/O data frames to form I/O matrix for (t-4)
io_matrix4 <- cbind(input_t1, input_t2, input_t3, input_t4, output)

#Remove rows with NA values
io_matrix4 <- na.omit(io_matrix4)

#normalize I/O matrix of(t-4) using z-score
io_matrix_norm4 = scale(io_matrix4)

#====================================================#

#combine I/O data frames to form I/O matrix for (t-7)
io_matrix7 <- cbind(input_t1, input_t2, input_t3, input_t4, input_t5, input_t6, input_t7, output)

#Remove rows with NA values
io_matrix7 <- na.omit(io_matrix7)

#normalize I/O matrix of(t-7) using z-score
io_matrix_norm7 = scale(io_matrix7)
#------------------------------------------------------------------------------#


#-------------------------------Model Training Phase---------------------------#
#(t-1) vector class models implementation

#Train/Test data split for (t-1) vector
train_data <- io_matrix_norm1[1:380, ]
test_data1 <- io_matrix_norm1[(381:nrow(io_matrix_norm1)), ]

# define input and output for nn
input <- io_matrix_norm1[1:380,1:1]
output <- io_matrix_norm1[1:380, 2]

#create data frame to feed into nn
io_matrix_df1 <- data.frame(input1 = input, output = output)

library(neuralnet)
#creating nn model function
nn_model <- function(hyperparameter) {
  model <- neuralnet(output ~ input1, data=io_matrix_df1, 
                     hidden=hyperparameter[[1]], act.fct=hyperparameter[[2]],
                     algorithm=hyperparameter[[3]], learningrate=hyperparameter[[4]], 
                     threshold=hyperparameter[[5]], err.fct="sse", 
                     linear.output=hyperparameter[[6]], rep=1, 
                     lifesign="full", stepmax=1e5)
  return(model)
}

#Define hyperparameters for 12 models
hyperparameters <- list(
  list(c(5,2), 'tanh', 'backprop', 1e-4, 0.5, TRUE),
  list(c(10,5), 'tanh', 'backprop', 1e-4, 0.5, FALSE),
  list(c(4,3), 'logistic','backprop', 1e-4, 0.5, TRUE),
  list(c(3,2), 'tanh','rprop+', 1e-5, 0.4, FALSE),
  list(c(6,1), 'logistic','rprop-', 1e-5, 0.4, TRUE),
  list(c(4,2), 'logistic','rprop+', 1e-6, 0.4, FALSE),
  list(c(3,1), 'tanh','sag', 1e-6, 0.3, TRUE),
  list(c(5,3), 'logistic','sag', 1e-6, 0.3, FALSE),
  list(c(2,2), 'tanh','sag', 1e-6, 0.3, TRUE),
  list(c(6,2), 'tanh', 'slr', 1e-7, 0.2, FALSE),
  list(c(5,1), 'logistic','slr', 1e-7, 0.2, TRUE),
  list(c(4,1), 'logistic','slr', 1e-7, 0.2, FALSE)
)


# Train a set of neural network models
nn_models1 <- list()

# Loop through each set of hyperparameters
for (i in seq_along(hyperparameters)) {
  nn_model_name <- paste("nn_", paste(hyperparameters[[i]], collapse=", "), sep="")
  print(paste("Training model:", nn_model_name))
  nn_models1[[nn_model_name]] <- nn_model(hyperparameters[[i]])
}

# Create an empty data frame to store the results
results_df1 <- data.frame(model = character(), RMSE = numeric(), 
                          MAE = numeric(), MAPE = numeric(), 
                          sMAPE = numeric())

feed_data <- data.frame(input1 <- test_data1[, 1:1])

# Loop through each model in nn_models
for (i in seq_along(nn_models1)) {
  
  #Get the predicted values for the test data
  predictions <- predict(nn_models1[[i]], feed_data)
  
  #Get the mean and standard deviation of the original output data
  output_mean <- mean(io_matrix_norm1)
  output_std <- sd(io_matrix_norm1)
  
  #Denormalize the predictions using Z-score
  predictions <- (predictions * output_std) + output_mean
  
  # Calculate the error metrics
  RMSE <- sqrt(mean((predictions - test_data1[, 2])^2))
  MAE <- mean(abs(predictions - test_data1[, 2]))
  MAPE <- mean(abs((test_data1[, 2] -  predictions)/test_data1[, 2])) * 100
  sMAPE <- mean(abs(test_data1[, 2] - predictions) / ((abs(test_data1[, 2]) + abs(predictions))/2)) * 100
  
  # Add the results to the data frame
  model_name <- names(nn_models1)[i]
  results_df1[i, "model"] <- model_name
  results_df1[i, "RMSE"] <- RMSE
  results_df1[i, "MAE"] <- MAE
  results_df1[i, "MAPE"] <- MAPE
  results_df1[i, "sMAPE"] <- sMAPE
}

# Print the results
print(results_df1)

# Identify the best model
best_model_1_name <- results_df1$model[which.min(results_df1$RMSE)]

# Best model
best_model_1 <- nn_models1[[best_model_1_name]]

# Plot the best model
plot(best_model_1)


#============================================================#
#(t-2) vector class models implementation

#Train/Test data split for (t-2) vector
  
train_data <- io_matrix_norm2[1:380, ]
test_data2 <- io_matrix_norm2[(381:nrow(io_matrix_norm2)), ]

# define input and output for nn
input <- io_matrix_norm2[1:380,1:2]
output <- as.vector(io_matrix_norm2[1:380, 3])

#create data frame to feed into nn
io_matrix_df2 <- data.frame(input1 = input[,1], input2 = input[,2], output = output)

#creating nn model function
nn_model <- function(hyperparameter) {
  model <- neuralnet(output ~ input1 + input2, data=io_matrix_df2, 
                     hidden=hyperparameter[[1]], act.fct=hyperparameter[[2]],
                     algorithm=hyperparameter[[3]], learningrate=hyperparameter[[4]], 
                     threshold=hyperparameter[[5]], err.fct="sse", 
                     linear.output=hyperparameter[[6]], rep=1, 
                     lifesign="full", stepmax=1e7)
  return(model)
}

#Define hyperparameters for 12 models
hyperparameters <- list(
  list(c(5,2), 'tanh', 'backprop', 1e-4, 0.5, TRUE),
  list(c(10,5), 'tanh', 'backprop', 1e-4, 0.5, FALSE),
  list(c(4,3), 'logistic','backprop', 1e-4, 0.5, TRUE),
  list(c(3,2), 'tanh','rprop+', 1e-5, 0.4, FALSE),
  list(c(6,1), 'logistic','rprop-', 1e-5, 0.4, TRUE),
  list(c(4,2), 'logistic','rprop+', 1e-6, 0.4, FALSE),
  list(c(3,1), 'tanh','sag', 1e-6, 0.3, TRUE),
  list(c(5,3), 'logistic','sag', 1e-6, 0.3, FALSE),
  list(c(2,2), 'tanh','sag', 1e-6, 0.3, TRUE),
  list(c(6,2), 'tanh', 'slr', 1e-7, 0.2, FALSE),
  list(c(5,1), 'logistic','slr', 1e-7, 0.2, TRUE),
  list(c(4,1), 'logistic','slr', 1e-7, 0.2, FALSE)
)

# Train a set of neural network models
nn_models2 <- list()

# Loop through each set of hyperparameters
for (i in seq_along(hyperparameters)) {
  nn_model_name <- paste("nn_", paste(hyperparameters[[i]], collapse=", "), sep="")
  print(paste("Training model:", nn_model_name))
  nn_models2[[nn_model_name]] <- nn_model(hyperparameters[[i]])
}

# Create an empty data frame to store the results
results_df2 <- data.frame(model = character(), RMSE = numeric(), 
                          MAE = numeric(), MAPE = numeric(), 
                          sMAPE = numeric())

# Loop through each model in nn_models
for (i in seq_along(nn_models2)) {
  
  #Get the predicted values for the test data
  predictions <- predict(nn_models2[[i]], test_data2[, 1:2])
  
  #Get the mean and standard deviation of the original output data
  output_mean <- mean(io_matrix_norm2)
  output_std <- sd(io_matrix_norm2)
  
  #Denormalize the predictions using Z-score
  predictions <- (predictions * output_std) + output_mean
  
  # Calculate the error metrics
  RMSE <- sqrt(mean((predictions - test_data2[, 3])^2))
  MAE <- mean(abs(predictions - test_data2[, 3]))
  MAPE <- mean(abs((test_data2[, 3] -  predictions)/test_data2[, 3])) * 100
  sMAPE <- mean(abs(test_data2[, 3] - predictions) / ((abs(test_data2[, 3]) + abs(predictions))/2)) * 100
  
  # Add the results to the data frame
  model_name <- names(nn_models2)[i]
  results_df2[i, "model"] <- model_name
  results_df2[i, "RMSE"] <- RMSE
  results_df2[i, "MAE"] <- MAE
  results_df2[i, "MAPE"] <- MAPE
  results_df2[i, "sMAPE"] <- sMAPE
}

# Print the results
print(results_df2)

# Identify the best model
best_model_2_name <- results_df2$model[which.min(results_df2$RMSE)]

# Best model
best_model_2 <- nn_models2[[best_model_2_name]]

# Plot the best model
plot(best_model_2)

#============================================================#
#(t-3) vector class models implementation

#Train/Test data split for (t-3) vector
train_data <- io_matrix_norm3[1:380, ]
test_data3 <- io_matrix_norm3[(381:nrow(io_matrix_norm3)), ]

# define input and output for nn
input <- io_matrix_norm3[1:380,1:3]
output <- as.vector(io_matrix_norm3[1:380, 4])

#create data frame to feed into nn
io_matrix_df3 <- data.frame(input1 = input[,1], input2 = input[,2], input3 = input[,3], output = output)

#creating nn model function
nn_model <- function(hyperparameter) {
  model <- neuralnet(output ~ input1 + input2 + input3, data=io_matrix_df3, 
                     hidden=hyperparameter[[1]], act.fct=hyperparameter[[2]],
                     algorithm=hyperparameter[[3]], learningrate=hyperparameter[[4]], 
                     threshold=hyperparameter[[5]], err.fct="sse", 
                     linear.output=hyperparameter[[6]], rep=1, 
                     lifesign="full", stepmax=1e7)
  return(model)
}
#Define hyperparameters for 12 models
hyperparameters <- list(
  list(c(5,2), 'tanh', 'backprop', 1e-4, 0.5, TRUE),
  list(c(10,5), 'tanh', 'backprop', 1e-4, 0.5, FALSE),
  list(c(4,3), 'logistic','backprop', 1e-4, 0.5, TRUE),
  list(c(3,2), 'tanh','rprop+', 1e-5, 0.4, FALSE),
  list(c(6,1), 'logistic','rprop-', 1e-5, 0.4, TRUE),
  list(c(4,2), 'logistic','rprop+', 1e-6, 0.4, FALSE),
  list(c(3,1), 'tanh','sag', 1e-6, 0.3, TRUE),
  list(c(5,3), 'logistic','sag', 1e-6, 0.3, FALSE),
  list(c(2,3), 'tanh','sag', 1e-6, 0.3, TRUE),
  list(c(6,2), 'tanh', 'slr', 1e-7, 0.2, FALSE),
  list(c(5,1), 'logistic','slr', 1e-7, 0.2, TRUE),
  list(c(4,1), 'logistic','slr', 1e-7, 0.2, FALSE)
)


# Train a set of neural network models
nn_models3 <- list()

# Loop through each set of hyperparameters
for (i in seq_along(hyperparameters)) {
  nn_model_name <- paste("nn_", paste(hyperparameters[[i]], collapse=", "), sep="")
  print(paste("Training model:", nn_model_name))
  nn_models3[[nn_model_name]] <- nn_model(hyperparameters[[i]])
}

# Create an empty data frame to store the results
results_df3 <- data.frame(model = character(), RMSE = numeric(), 
                          MAE = numeric(), MAPE = numeric(), 
                          sMAPE = numeric())

# Loop through each model in nn_models
for (i in seq_along(nn_models3)) {
  
  #Get the predicted values for the test data
  predictions <- predict(nn_models3[[i]], test_data3[, 1:3])
  
  #Get the mean and standard deviation of the original output data
  output_mean <- mean(io_matrix_norm3)
  output_std <- sd(io_matrix_norm3)
  
  #Denormalize the predictions using Z-score
  predictions <- (predictions * output_std) + output_mean
  
  # Calculate the error metrics
  RMSE <- sqrt(mean((predictions - test_data3[, 4])^2))
  MAE <- mean(abs(predictions - test_data3[, 4]))
  MAPE <- mean(abs((test_data3[, 4] -  predictions)/test_data3[, 4])) * 100
  sMAPE <- mean(abs(test_data3[, 4] - predictions) / ((abs(test_data3[, 4]) + abs(predictions))/2)) * 100
  
  # Add the results to the data frame
  model_name <- names(nn_models3)[i]
  results_df3[i, "model"] <- model_name
  results_df3[i, "RMSE"] <- RMSE
  results_df3[i, "MAE"] <- MAE
  results_df3[i, "MAPE"] <- MAPE
  results_df3[i, "sMAPE"] <- sMAPE
}

# Print the results
print(results_df3)

# Identify the best model
best_model_3_name <- results_df3$model[which.min(results_df3$RMSE)]

# Best model
best_model_3 <- nn_models3[[best_model_3_name]]

# Plot the best model
plot(best_model_3)

#============================================================#
#(t-4) vector class models implementation

#Train/Test data split for (t-4) vector
train_data <- io_matrix_norm4[1:380, ]
test_data4 <- io_matrix_norm4[(381:nrow(io_matrix_norm4)), ]

# define input and output for nn
input <- io_matrix_norm4[1:380,1:4]
output <- as.vector(io_matrix_norm4[1:380, 5])

#create data frame to feed into nn
io_matrix_df4 <- data.frame(input1 = input[,1], input2 = input[,2], input3 = input[,3], 
                           input4 = input[,4],
                           output = output)
#creating nn model function
nn_model <- function(hyperparameter) {
  model <- neuralnet(output ~ input1 + input2 + input3 + input4, data=io_matrix_df4, 
                     hidden=hyperparameter[[1]], act.fct=hyperparameter[[2]],
                     algorithm=hyperparameter[[3]], learningrate=hyperparameter[[4]], 
                     threshold=hyperparameter[[5]], err.fct="sse", 
                     linear.output=hyperparameter[[6]], rep=1, 
                     lifesign="full", stepmax=1e7)
  return(model)
}

#Define hyperparameters for 12 models
hyperparameters <- list(
  list(c(5,2), 'tanh', 'backprop', 1e-4, 0.5, TRUE),
  list(c(10,5), 'tanh', 'backprop', 1e-4, 0.5, FALSE),
  list(c(4,3), 'logistic','backprop', 1e-4, 0.5, TRUE),
  list(c(3,2), 'tanh','rprop+', 1e-5, 0.4, FALSE),
  list(c(6,1), 'logistic','rprop-', 1e-5, 0.4, TRUE),
  list(c(4,2), 'logistic','rprop+', 1e-6, 0.4, FALSE),
  list(c(3,1), 'tanh','sag', 1e-6, 0.5, TRUE),
  list(c(5,3), 'logistic','sag', 1e-6, 0.5, FALSE),
  list(c(2,3), 'tanh','sag', 1e-6, 0.5, TRUE),
  list(c(6,2), 'tanh', 'slr', 1e-7, 0.5, FALSE),
  list(c(5,1), 'logistic','slr', 1e-7, 0.5, TRUE),
  list(c(4,1), 'logistic','slr', 1e-7, 0.5, FALSE)
)


# Train a set of neural network models
nn_models4 <- list()

# Loop through each set of hyperparameters
for (i in seq_along(hyperparameters)) {
  nn_model_name <- paste("nn_", paste(hyperparameters[[i]], collapse=", "), sep="")
  print(paste("Training model:", nn_model_name))
  nn_models4[[nn_model_name]] <- nn_model(hyperparameters[[i]])
}

# Create an empty data frame to store the results
results_df4 <- data.frame(model = character(), RMSE = numeric(), 
                          MAE = numeric(), MAPE = numeric(), 
                          sMAPE = numeric())

# Loop through each model in nn_models
for (i in seq_along(nn_models4)) {
  
  #Get the predicted values for the test data
  predictions <- predict(nn_models4[[i]], test_data4[, 1:4])
  
  #Get the mean and standard deviation of the original output data
  output_mean <- mean(io_matrix_norm4)
  output_std <- sd(io_matrix_norm4)
  
  #Denormalize the predictions using Z-score
  predictions <- (predictions * output_std) + output_mean
  
  # Calculate the error metrics
  RMSE <- sqrt(mean((predictions - test_data4[, 5])^2))
  MAE <- mean(abs(predictions - test_data4[, 5]))
  MAPE <- mean(abs((test_data4[, 5] -  predictions)/test_data4[, 5])) * 100
  sMAPE <- mean(abs(test_data4[, 5] - predictions) / ((abs(test_data4[, 5]) + abs(predictions))/2)) * 100
  
  # Add the results to the data frame
  model_name <- names(nn_models4)[i]
  results_df4[i, "model"] <- model_name
  results_df4[i, "RMSE"] <- RMSE
  results_df4[i, "MAE"] <- MAE
  results_df4[i, "MAPE"] <- MAPE
  results_df4[i, "sMAPE"] <- sMAPE
}

# Print the results
print(results_df4)

# Identify the best model
best_model_4_name <- results_df4$model[which.min(results_df4$RMSE)]

# Best model
best_model_4 <- nn_models4[[best_model_4_name]]

# Plot the best model
plot(best_model_4)


#============================================================#
#(t-7) vector class models implementation
#Train/Test data split for (t-7) vector
train_data7 <- io_matrix_norm7[1:380, ]
test_data7 <- io_matrix_norm7[(381:nrow(io_matrix_norm7)), ]

# define input and output for nn
input <- io_matrix_norm7[1:380,1:7]
output <- as.vector(io_matrix_norm7[1:380, 8])

#create data frame to feed into nn
io_matrix_df7 <- data.frame(input1 = input[,1], input2 = input[,2], input3 = input[,3], 
                           input4 = input[,4], input5 = input[,5], input6 = input[,6], input7 = input[,7], 
                           output = output)

#creating nn model function
nn_model <- function(hyperparameter) {
  model <- neuralnet(output ~ input1 + input2 + input3 + input4 + input5 + input6 + input7, data=io_matrix_df7, 
                     hidden=hyperparameter[[1]], act.fct=hyperparameter[[2]],
                     algorithm=hyperparameter[[3]], learningrate=hyperparameter[[4]], 
                     threshold=hyperparameter[[5]], err.fct="sse", 
                     linear.output=hyperparameter[[6]], rep=1, 
                     lifesign="full", stepmax=1e5)
  return(model)
}

#Define hyperparameters for 12 models
hyperparameters <- list(
  list(c(5,2), 'tanh', 'backprop', 1e-4, 0.5, TRUE),
  list(c(10,5), 'tanh', 'backprop', 1e-4, 0.5, FALSE),
  list(c(4,3), 'logistic','backprop', 1e-4, 0.5, TRUE),
  list(c(3,2), 'tanh','rprop+', 1e-5, 0.4, FALSE),
  list(c(6,1), 'logistic','rprop-', 1e-5, 0.4, TRUE),
  list(c(4,2), 'logistic','rprop+', 1e-6, 0.4, FALSE),
  list(c(3,1), 'tanh','sag', 1e-6, 0.3, TRUE),
  list(c(5,3), 'logistic','sag', 1e-6, 0.3, FALSE),
  list(c(2,3), 'tanh','sag', 1e-6, 0.3, TRUE),
  list(c(6,2), 'tanh', 'slr', 1e-7, 0.2, FALSE),
  list(c(5,1), 'logistic','slr', 1e-7, 0.2, TRUE),
  list(c(4,1), 'logistic','slr', 1e-7, 0.2, FALSE)
)

#Train a set of neural network models
nn_models7 <- list()

#Loop through each set of hyperparameters
for (i in seq_along(hyperparameters)) {
  nn_model_name <- paste("nn_", paste(hyperparameters[[i]], collapse=", "), sep="")
  print(paste("Training model:", nn_model_name))
  nn_models7[[nn_model_name]] <- nn_model(hyperparameters[[i]])
}

# Create an empty data frame to store the results
results_df7 <- data.frame(model = character(), RMSE = numeric(), 
                          MAE = numeric(), MAPE = numeric(), sMAPE = numeric())

# Loop through each model in nn_models7
for (i in seq_along(nn_models7)) {
  
  #Get the predicted values for the test data
  predictions <- predict(nn_models7[[i]], test_data7[, 1:7])
  
  #Get the mean and standard deviation of the original output data
  output_mean <- mean(io_matrix_norm7[, 8])
  output_std <- sd(io_matrix_norm7[, 8])
  
  #Denormalize the predictions using Z-score
  predictions <- (predictions * output_std) + output_mean
  
  # Calculate the error metrics
  RMSE <- sqrt(mean((predictions - test_data7[, 8])^2))
  MAE <- mean(abs(predictions - test_data7[, 8]))
  MAPE <- mean(abs((test_data7[, 8] - predictions)/test_data7[, 8])) * 100
  sMAPE <- mean(abs(test_data7[, 8] - predictions) / ((abs(test_data7[, 8]) + abs(predictions))/2)) * 100
  
  # Add the results to the data frame
  results_df7[i, "model"] <- names(nn_models7)[i]
  results_df7[i, "RMSE"] <- RMSE
  results_df7[i, "MAE"] <- MAE
  results_df7[i, "MAPE"] <- MAPE
  results_df7[i, "sMAPE"] <- sMAPE
}

# Print the results
print(results_df7)

# Identify the best model
best_model_7_name <- results_df7$model[which.min(results_df7$RMSE)]

# Best model
best_model_7 <- nn_models7[[best_model_7_name]]

# Plot the best model
plot(best_model_7)

