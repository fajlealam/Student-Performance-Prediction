# Student Performance Prediction (GR-2)

A beginner-friendly BCA machine-learning project that predicts expected final marks using attendance, study hours, assignment score, class participation, and previous marks.

## Features
- Sample student dataset (1,000 **synthetic** records)
- Data cleaning and preprocessing
- Exploratory data analysis through charts
- Train/test split (80% / 20%)
- Linear Regression and Random Forest comparison
- Evaluation using MAE, RMSE, and R²
- Saved best model
- Streamlit interface for interactive predictions
- Feature-importance display

> **Important:** The included data is synthetically generated for learning and demonstration. It does not represent real students, and the model's scores are not evidence of real-world predictive accuracy. For a real study, use a suitable verified dataset and follow your institution's privacy rules.

## Requirements
- Python 3.9 or newer
- Internet connection for installing packages (not needed after installation)

## Setup (Windows)
1. Extract the ZIP file.
2. Open Command Prompt in the extracted `student_performance_prediction` folder.
3. Create a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Activate it:

   ```bash
   venv\Scripts\activate
   ```

5. Install packages:

   ```bash
   pip install -r requirements.txt
   ```

6. Train and evaluate the models:

   ```bash
   python train_model.py
   ```

7. Start the web application:

   ```bash
   streamlit run app.py
   ```

8. A browser tab will open. Enter student details in the sidebar and click **Predict performance**.

## Setup (macOS / Linux)
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Dataset columns
| Column | Meaning |
|---|---|
| student_id | Anonymous sample student ID |
| attendance_percentage | Attendance from 0 to 100 |
| study_hours_per_day | Average daily study hours |
| assignment_score | Assignment score out of 100 |
| class_participation | Participation rating from 1 to 10 |
| previous_marks | Previous exam marks out of 100 |
| final_marks | Target value: final exam marks out of 100 |

## Project workflow
1. Load the CSV dataset with Pandas.
2. Convert columns to numeric values and handle missing feature values using median imputation.
3. Split data into training and testing portions.
4. Train Linear Regression and Random Forest regression models.
5. Compare test-set MAE, RMSE, and R²; save the model with lower MAE.
6. Use the saved model to predict marks from the Streamlit interface.

## Evaluation metrics
- **MAE:** Average absolute difference between predicted and actual marks; lower is better.
- **RMSE:** Penalizes larger prediction errors more strongly; lower is better.
- **R²:** Measures how much target variation is explained by the model; closer to 1 is generally better.

## Files
- `app.py` — Streamlit user interface
- `train_model.py` — preprocessing, training, evaluation, and model saving
- `data/student_performance.csv` — sample synthetic dataset
- `models/` — trained model is created here after running the training script

## Limitations and ethical use
Predictions are estimates and can be wrong. Attendance or study hours should not be used alone to judge a student. Real student data should be anonymized, collected with appropriate permission, and used to support students rather than label or penalize them.
