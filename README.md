# Breast_Cancer_Detection_Using_NN

🧠 Breast Cancer Detection using Neural Networks

A deep learning project that classifies breast tumors as Malignant or Benign using a fully connected Artificial Neural Network (ANN) trained on diagnostic medical features.

The system takes numerical tumor measurements and predicts cancer type with 96.5% accuracy.

Think of it as:
data → scaling → neural network → diagnosis.

Simple. Deterministic. Effective. Like good engineering should be.

🚀 Problem Statement

Early breast cancer detection dramatically improves survival rates.

Manual diagnosis can be subjective and time-consuming.
This project builds a machine learning classifier to assist diagnosis using measurable cellular features.

Goal:
Automatically predict:

0 → Malignant

1 → Benign

📂 Dataset

Breast Cancer Wisconsin Diagnostic Dataset

Loaded directly from sklearn.datasets.

Contains:

569 samples

30 numerical features

Features include radius, texture, perimeter, area, smoothness, etc.

Binary classification target

🧰 Tech Stack

Python

NumPy

Pandas

Matplotlib / Seaborn

Scikit-learn

TensorFlow / Keras

⚙️ Workflow
Data Processing

Dataset loaded using sklearn

Converted to Pandas DataFrame

Train–test split

Standardization using StandardScaler

Mean = 0

Std = 1

Helps neural network converge faster

Feature scaling is critical here.
Without it, training behaves like a drunk elephant.

Model Architecture

Fully Connected Neural Network:

Input (30 features)
↓
Flatten
↓
Dense (20 neurons, ReLU)
↓
Dense (2 neurons, Sigmoid output)

Implementation:

keras.Sequential([
    keras.layers.Flatten(input_shape=(30,)),
    keras.layers.Dense(20, activation='relu'),
    keras.layers.Dense(2, activation='sigmoid')
])

Training Configuration

Optimizer → Adam

Loss → sparse_categorical_crossentropy

Epochs → 20

Validation split → 20%

Metric → Accuracy

Sparse categorical loss is used because labels are integers (0,1).

📊 Results

Test set performance:

Metric	Value
Loss	0.091
Accuracy	96.49%
Interpretation

Correct predictions for ~96 out of 100 patients

Low loss indicates confident predictions

Suitable for medical screening assistance

Validation curves show:

Decreasing loss

Increasing accuracy

No major overfitting

Stable training. Clean convergence. Chef’s kiss.

📈 Visualizations

The notebook includes:

Training vs Validation Loss curve

Training vs Validation Accuracy curve

Used to monitor overfitting and learning behavior.

🔮 Predictive System (Single Patient Inference)

The model also supports real-time prediction for new data:

Steps:

Input 30 tumor features

Scale using trained scaler

Run through model

Output class label

Example output:

Malignant / Benign


This simulates how a real clinical decision-support system would work.

▶️ How to Run

Clone:

git clone <repo-url>
cd breast-cancer-nn


Install dependencies:

pip install tensorflow scikit-learn pandas matplotlib seaborn


Run notebook:

jupyter notebook Breast_Cancer_Using_NN.ipynb

🧠 Key Learnings

This project demonstrates:

Neural networks for tabular medical data

Importance of feature scaling

Binary classification with sparse labels

Model evaluation on unseen data

Building end-to-end prediction pipelines

