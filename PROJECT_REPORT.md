# GR-2: Student Performance Prediction

## 1. Introduction
Student performance can be associated with academic and behavioural factors such as attendance, study time, assignment results, participation, and previous marks. This project demonstrates how regression algorithms can estimate a student's final marks from these inputs.

## 2. Problem statement
To develop a simple machine-learning application that predicts expected final marks and identifies which input features are most influential in the trained model.

## 3. Objectives
- Prepare a structured student-performance dataset.
- Clean and preprocess the data.
- Explore the data using descriptive statistics and charts.
- Train and compare regression algorithms.
- Evaluate predictions on unseen test data.
- Build a simple interface for making predictions.

## 4. Dataset
The supplied CSV contains 1,000 synthetically generated records. The input features are attendance percentage, study hours per day, assignment score, class participation, and previous marks. Final marks are the target variable. Synthetic data is used only to make the project runnable; it must not be presented as actual student observations.

## 5. Methodology
1. Read the CSV using Pandas.
2. Convert model columns to numeric format and remove rows with a missing target.
3. Use median imputation for missing input values.
4. Split the dataset into 80% training and 20% testing data.
5. Train Linear Regression and Random Forest Regression.
6. Compare models using MAE, RMSE, and R².
7. Save the model with the lower test MAE.
8. Load the saved model in a Streamlit application.

## 6. Algorithms
**Linear Regression:** Estimates a continuous target by learning a weighted relationship between input variables and final marks.

**Random Forest Regression:** Combines predictions from multiple decision trees to model more complex relationships.

## 7. Evaluation
The training script prints test-set MAE, RMSE, and R² for both algorithms. Record the actual values produced on your computer here after running `python train_model.py`; do not invent or copy performance values from an unrelated run.

| Algorithm | MAE | RMSE | R² |
|---|---:|---:|---:|
| Linear Regression | Run script | Run script | Run script |
| Random Forest | Run script | Run script | Run script |

## 8. Expected outcome
The application accepts student-related inputs and displays predicted final marks, a broad performance category, dataset summaries, and feature-importance information.

## 9. Limitations
- The sample data is synthetic and may encode artificial relationships.
- Model quality on this sample does not establish accuracy on real students.
- The model estimates association, not causation.
- Important influences such as health, teaching quality, family circumstances, and exam difficulty are not represented.

## 10. Conclusion
This project demonstrates a complete introductory machine-learning workflow, from dataset preparation and exploratory analysis to model evaluation and an interactive prediction interface. Future work can use a validated real dataset, add more relevant features, and test model fairness and generalization.
