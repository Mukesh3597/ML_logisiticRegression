📌 Logistic Regression From Scratch
🚀 Project Overview

This project implements Logistic Regression from scratch using NumPy without using any built-in machine learning model for training.

The model predicts whether a user will purchase a product based on:

Gender

Age

Estimated Salary

📊 Dataset

Dataset: Logistic Regression Dataset

Features:

Gender

Age

EstimatedSalary

Target:

Purchased (0 or 1)

⚙️ Steps Performed

1️⃣ Data Loading
2️⃣ Data Cleaning (Dropped User ID)
3️⃣ Feature Encoding (Gender → 0/1)
4️⃣ Train-Test Split
5️⃣ Feature Scaling (Age, Salary)
6️⃣ Logistic Regression Implementation (from scratch)
7️⃣ Prediction
8️⃣ Model Evaluation

🧠 Algorithm Used

Logistic Regression

Sigmoid Function

Gradient Descent

📈 Results

Accuracy: 86.25%

Example Output:

Accuracy: 0.8625
First 10 Predictions: [0 1 0 1 0 0 1 0 0 0]
First 10 Actual:      [0 1 0 1 0 0 1 0 0 0]
🔢 Model Parameters
Weights: [-0.0200, 1.1861, 0.7113]
Bias: -0.6144
🛠️ Tech Stack

Python

NumPy

Pandas

Scikit-learn

Matplotlib / Seaborn

📂 Project Structure
LOGISITIC/
│
├── data/
│   └── logisiticRegression.csv
│
├── scr/
│   └── logistic_regression.py
│
├── main.py
└── requirements.txt
▶️ How to Run
pip install -r requirements.txt
python main.py
💡 Key Learning

Implemented ML algorithm from scratch

Understood gradient descent deeply

Learned importance of feature scaling

Practiced end-to-end ML pipeline

🔗 Author

Mukesh
BCA Student | Aspiring ML Engineer