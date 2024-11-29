# Load necessary libraries
library(tidyverse)
library(dplyr)
library(randomForest)
library(pROC)
library(ggplot2)

# Load the dataset
data <- read.csv("D:\\r_diabetes\\diabetes\\diabetes_prediction_dataset.csv") 

# Preprocess the data
data$bmi[is.na(data$bmi)] <- mean(data$bmi, na.rm = TRUE)
data$HbA1c_level[is.na(data$HbA1c_level)] <- mean(data$HbA1c_level, na.rm = TRUE)
data$blood_glucose_level[is.na(data$blood_glucose_level)] <- mean(data$blood_glucose_level, na.rm = TRUE)
data$smoking_h[is.na(data$smoking_h)] <- "No Info"
data$gender <- as.numeric(factor(data$gender, levels = c("Female", "Male")))
data$diabetes <- as.factor(data$diabetes)
num_cols <- c("age", "bmi", "HbA1c_level", "blood_glucose_level")
data[num_cols] <- scale(data[num_cols])
data <- na.omit(data)

# Split data into training and testing sets
set.seed(123)
sample_index <- sample(1:nrow(data), size = 0.3 * nrow(data))
train <- data[sample_index, ]
test <- data[-sample_index, ]

# --- Logistic Regression Model ---
model_glm <- glm(diabetes ~ ., data = train, family = "binomial")
prob_glm <- predict(model_glm, test, type = "response")
roc_glm <- roc(test$diabetes, prob_glm)
auc_glm <- auc(roc_glm)

# --- Random Forest Model ---
model_rf <- randomForest(diabetes ~ ., data = train, ntree = 50, mtry = 3)
prob_rf <- predict(model_rf, test, type = "prob")[,2]
roc_rf <- roc(test$diabetes, prob_rf)
auc_rf <- auc(roc_rf)

# --- Calculate Metrics ---
# For GLM model
pred_glm <- ifelse(prob_glm > 0.5, 1, 0)
conf_matrix_glm <- table(pred_glm, test$diabetes)

accuracy_glm <- sum(diag(conf_matrix_glm)) / sum(conf_matrix_glm)
precision_glm <- conf_matrix_glm[2,2] / sum(conf_matrix_glm[2,])
recall_glm <- conf_matrix_glm[2,2] / sum(conf_matrix_glm[2,1], conf_matrix_glm[2,2])
fscore_glm <- 2 * (precision_glm * recall_glm) / (precision_glm + recall_glm)

# For Random Forest model
pred_rf <- ifelse(prob_rf > 0.5, 1, 0)
conf_matrix_rf <- table(pred_rf, test$diabetes)

accuracy_rf <- sum(diag(conf_matrix_rf)) / sum(conf_matrix_rf)
precision_rf <- conf_matrix_rf[2,2] / sum(conf_matrix_rf[2,])
recall_rf <- conf_matrix_rf[2,2] / sum(conf_matrix_rf[2,1], conf_matrix_rf[2,2])
fscore_rf <- 2 * (precision_rf * recall_rf) / (precision_rf + recall_rf)

# --- Print Metrics ---
cat("GLM Accuracy:", accuracy_glm, "\n")
cat("GLM Precision:", precision_glm, "\n")
cat("GLM Recall:", recall_glm, "\n")
cat("GLM F-Score:", fscore_glm, "\n")

cat("Random Forest Accuracy:", accuracy_rf, "\n")
cat("Random Forest Precision:", precision_rf, "\n")
cat("Random Forest Recall:", recall_rf, "\n")
cat("Random Forest F-Score:", fscore_rf, "\n")

# --- Create Comparison Plot ---
# Store metrics in a data frame
metrics <- data.frame(
  Model = c("GLM", "Random Forest"),
  Accuracy = c(accuracy_glm, accuracy_rf),
  Precision = c(precision_glm, precision_rf),
  Recall = c(recall_glm, recall_rf),
  F_Score = c(fscore_glm, fscore_rf)
)

# Reshape the data to a long format for plotting
metrics_long <- metrics %>%
  pivot_longer(cols = c(Accuracy, Precision, Recall, F_Score), 
               names_to = "Metric", 
               values_to = "Value")

# Plot comparison of accuracy, precision, recall, and F-score
ggplot(metrics_long, aes(x = Model, y = Value, fill = Metric)) +
  geom_bar(stat = "identity", position = "dodge") +
  labs(title = "Comparison of Accuracy, Precision, Recall, and F-Score",
       x = "Model", y = "Value") +
  scale_fill_manual(values = c("lightblue", "lightgreen", "lightcoral", "lightyellow")) +
  theme_minimal()
