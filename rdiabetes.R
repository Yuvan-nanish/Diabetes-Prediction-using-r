
# Load the dataset
library(tidyverse)
library(dplyr)
library(caret)

# Load the dataset# Using double backslashes
data <- read.csv("D:\\r_diabetes\\diabetes\\diabetes_prediction_dataset.csv")

# View the structure of the dataset
str(data)
summary(data)

# Check for missing values
colSums(is.na(data))

# Fill missing values
data$bmi[is.na(data$bmi)] <- mean(data$bmi, na.rm = TRUE)
data$HbA1c_level[is.na(data$HbA1c_level)] <- mean(data$HbA1c_level, na.rm = TRUE)
data$blood_glucose_level[is.na(data$blood_glucose_level)] <- mean(data$blood_glucose_level, na.rm = TRUE)
data$smoking_h[is.na(data$smoking_h)] <- "No Info"

# Encode categorical variables
data$gender <- as.numeric(factor(data$gender, levels = c("Female", "Male")))
data$smoking_h <- as.factor(data$smoking_h)

# Normalize numerical columns
num_cols <- c("age", "bmi", "HbA1c_level", "blood_glucose_level")
data[num_cols] <- scale(data[num_cols])

# Load libraries for visualization
library(ggplot2)

# Count plot for diabetes
ggplot(data, aes(x = as.factor(diabetes))) +
  geom_bar(fill = "skyblue") +
  labs(title = "Diabetes Count", x = "Diabetes", y = "Count")

# Boxplot for age by diabetes status
ggplot(data, aes(x = as.factor(diabetes), y = age)) +
  geom_boxplot(fill = "orange") +
  labs(title = "Age by Diabetes Status", x = "Diabetes", y = "Age")

# Boxplot for BMI by diabetes status
ggplot(data, aes(x = as.factor(diabetes), y = bmi)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "BMI by Diabetes Status", x = "Diabetes", y = "BMI")

# Boxplot for HbA1c level by diabetes status
ggplot(data, aes(x = as.factor(diabetes), y = HbA1c_level)) +
  geom_boxplot(fill = "lightblue") +
  labs(title = "HbA1c Level by Diabetes Status", x = "Diabetes", y = "HbA1c Level")

# Boxplot for blood glucose level by diabetes status
ggplot(data, aes(x = as.factor(diabetes), y = blood_glucose_level)) +
  geom_boxplot(fill = "pink") +
  labs(title = "Blood Glucose Level by Diabetes Status", x = "Diabetes", y = "Blood Glucose Level")

# Split the data into training and test sets
set.seed(42)
trainIndex <- createDataPartition(data$diabetes, p = 0.8, list = FALSE)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Train logistic regression model
model <- glm(diabetes ~ ., data = trainData, family = binomial)

# Predict on test data
predictions <- predict(model, testData, type = "response")
predicted_classes <- ifelse(predictions > 0.5, 1, 0)

# Evaluate the model
conf_matrix <- table(Predicted = predicted_classes, Actual = testData$diabetes)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
cat("Accuracy:", accuracy, "\n")

# Detailed evaluation
library(caret)
confusionMatrix(as.factor(predicted_classes), as.factor(testData$diabetes))

