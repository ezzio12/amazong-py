# sharkpy/test.py

from sharkpy.core import Shark
import pandas as pd
import os
import numpy as np

# Step 0: Load data
data = pd.read_csv("data/Default.csv")
if "Unnamed: 0" in data.columns:
    data.drop(columns=["Unnamed: 0"], inplace=True)  # Remove unnamed columns if they exist

X = data.drop(columns=["default"])
y = data["default"]

# Step 1: Train Sharky
shark = Shark()
shark.learn(
    data=data,
    project_name="DefaultPrediction",
    target_column="default",
    detailed_stats=True)

# Step 2: Different Prediction Approaches
print("\n" + "="*60)
print("🦈 SHARKPY PREDICTION DEMO")
print("="*60)

# New Approach 1: No arguments - uses representative samples
print("\n📊 Approach 1: Default Prediction (Representative Samples)")
shark.predict()  # No arguments needed!

# New Approach 2: Baseline prediction
print("\n📊 Approach 2: Baseline Prediction")
baseline_pred = shark.predict_baseline()
print(f"Baseline prediction: {baseline_pred}")

# New Approach 3: Specific customer scenarios
print("\n📊 Approach 3: Specific Customer Scenarios")
scenarios = [
    {'student': 'No', 'balance': 1000, 'income': 30000},   # Low-risk customer
    {'student': 'Yes', 'balance': 5000, 'income': 80000},  # High-risk customer
    {'student': 'No', 'balance': 0, 'income': 20000}       # No balance customer
]

for i, scenario in enumerate(scenarios, 1):
    pred = shark.predict(scenario)
    print(f"Scenario {i}: {scenario}")
    print(f"   Prediction: {pred}")

# Step 3: Report (with export)
print("\n" + "="*60)
print("📈 PERFORMANCE REPORT")
print("="*60)
os.makedirs("reports", exist_ok=True)
shark.report(export_path="reports")

# Step 4: Explain
print("\n" + "="*60)
print("🔍 MODEL EXPLANATION")
print("="*60)
shark.explain()

# Step 5: Plot results
print("\n" + "="*60)
print("📊 CLASSIFICATION PLOTS")
print("="*60)
shark.plot(None, None, kind="confusion_matrix")

shark.plot(None, None, kind="roc")
