#------------------------------ Load necessary libraries-----------------------
library(neuralnet)
library(ggplot2)
library(readxl)
library(reshape2)
library(gridExtra)
------------------------------------------------------------------------------------
#------------------------------------ Load the data-------------------------------
exchange_data <- read_excel("C:/Users/venuk/Desktop/ML/CW/ExchangeUSD (2).xlsx")
str(exchange_data)
# Check for missing values in the exchange data 
sum(is.na(exchange_data))

# Extract the USD/EUR exchange rates
exchange_rates <- exchange_data$`USD/EUR`

# Split data into training and testing sets
training_data <- exchange_rates[1:400]
testing_data <- exchange_rates[401:500]

#--------------------------------Pre Processing ------------------------------

# Normalize the data using min-max normalization
normalize <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))
}

#-------------------- Normalize training and testing data-----------------
 training_data_norm<- normalize(training_data)
 testing_data_norm<- normalize(testing_data)
-------------------------------------------------------------------------------------

   
   
# -----------------------Finding AR  and creating maties -----------
 
# Function : to find AR Lags and creating the I/O Matrix
 
 io_matrix_creation_fct <- function(data, lag) {
   n <- length(data) - lag
   if(n <= 0) stop("Not enough data for the given lag.")
   
   input_matrix <- matrix(nrow = n, ncol = lag)
   output_matrix <- matrix(nrow = n, ncol = 1)
   
   for (i in 1:n) {
     input_matrix[i, ] <- data[i:(i + lag - 1)]
     output_matrix[i, ] <- data[i + lag]
   }
   
   # Assign column names to input_matrix
   colnames(input_matrix) <- paste0("lag", 1:lag)
   colnames(output_matrix) <- "output"
   
   return(list(input = input_matrix, output = output_matrix))
 }
 
 
 #-------------- Experiment with different lags (in this case, we change values to test)-------------
 lag_num <- c(2)
 
 # Create input-output matrices for training data
 tr_io_matrices <- lapply(lag_num, function(lag) io_matrix_creation_fct(training_data_norm, lag))
 # Create input-output matrices for testing data
 tst_io_matrices <- lapply(lag_num, function(lag) io_matrix_creation_fct(testing_data_norm, lag))
 
 # Check the structure of input-output matrices
 head(tr_io_matrices[[1]]$input)
 head(tr_io_matrices[[1]]$output)
 
 #-------------------------------------------------------------------------------------------------------
   
   
   #------------------------------------Neural Networking------------------------------------------
 
# FUNCTION : NN TRAINING
 # Function to train a neural network model
 train_neural_network <- function(input_data, output_data, hidden_layers, activation_function, learning_rate) {
   # Define formula
   formula <- as.formula(paste("output ~", paste(colnames(input_data), collapse = " + "), sep=""))
   
   # Define model structure
   model <- neuralnet(formula, data = cbind(input_data, output_data), 
                      hidden = hidden_layers,
                      linear.output = TRUE, 
                      act.fct = activation_function,
                      learningrate = learning_rate,
                      algorithm = "sag")
   
   
   return( model)
 }
 start_time <- Sys.time()
 # Train the neural network model
 mlp_model <- train_neural_network(
   input_data = tr_io_matrices[[1]]$input,
   output_data = tr_io_matrices[[1]]$output,
   hidden_layers = c(2,5),  # the amount of hidden layers
   activation_function = "tanh",  # Activation functions u can add any to test
   learning_rate = 0.01  # Learning rate
 )

 # Evaluate training time
 
 # Train one-hidden layer model
 end_time <- Sys.time()
 training_time <- end_time - start_time

 
 #----------------------------- FUNCTION : Denormalizing -------------------------------------
 
 # Denormalize data (reverse min-max scaling) cus we used min max scaling at the begining
 denormalize <- function(x, original_data) {
   minOriginal <- min(original_data)
   maxOriginal <- max(original_data)
   denormalizedData <- x * (maxOriginal - minOriginal) + minOriginal
   return(denormalizedData)
 }
 
 #---------------------------------------prdictiing-------------------------------
 predictions <- predict(mlp_model, tst_io_matrices[[1]]$input) #model is the mlp model
 
 #------------------------ Denormalize predictions and test output-----------------------
 denormalized_predictions <- denormalize(predictions, testing_data)#denormalize the prediction
 denormalized_test_output <- denormalize(tst_io_matrices[[1]]$output, testing_data)#denormalize the test outouts
 
 # Access the weights from the mlp_model object
 weights <- mlp_model$weights
 
 # Calculate the total number of weight parameters
 total_weights <- sum(sapply(weights, length))
 
 # Print the total number of weight parameters
 print(paste("Total number of weight parameters:", total_weights))
 # Calculate RMSE
 rmse <- sqrt(mean((denormalized_predictions - denormalized_test_output)^2))
 
 # Calculate MAE
 mae <- mean(abs(denormalized_predictions - denormalized_test_output))
 
 # Calculate MAPE
 mape <- mean(abs((denormalized_predictions - denormalized_test_output) / denormalized_test_output)) * 100
 
 # Calculate sMAPE
 smape <- mean(2 * abs(denormalized_predictions - denormalized_test_output) / (abs(denormalized_predictions) + abs(denormalized_test_output))) * 100
 
 # Print the evaluation metrics
 cat("RMSE:", rmse, "\n")
 cat("MAE:", mae, "\n")
 cat("MAPE:", mape, "%\n")
 cat("sMAPE:", smape, "%\n")
 
 # ----------------------------Plot predicted vs actual values---------------------------------
 
 plot(denormalized_test_output, type = "l", col = "darkgreen", xlab = "Index", ylab = "Value", main = "Predicted vs Actual Values")
 lines(denormalized_predictions, col = "red")
 legend("topleft", legend = c("Actual", "Predicted"), col = c("darkgreen", "red"), lty = 1)
 
 # ------------------------Visualize the neural network structure-------------------------------
 plot(mlp_model)
 summary(mlp_model)
 
 # -------------------Plot predicted vs actual values (scatter plot)---------------------------
 plot(denormalized_test_output, denormalized_predictions, col = "blue", 
      xlab = "Desired Output", ylab = "Predicted Output", 
      main = "Predicted vs Actual Values (Scatter Plot)")
 abline(0, 1, col = "red", lty = 2)  # Add a diagonal line for reference
 legend("topleft", legend = "Ideal Prediction", col = "red", lty = 2)
 
 
 #------------------------------------training time------------------------------------
 print(paste("Training Time:", training_time))

 
 
 
 
 
 
 
 