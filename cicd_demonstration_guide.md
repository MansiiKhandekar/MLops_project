# CI/CD Pipeline Demonstration Guide

This document captures the exact steps that were taken to initialize the Git repository locally and the simulated commits that represent the "Source Code Change" and "Dataset Change" scenarios. You can use this for your college viva to demonstrate your Git history and explain what happens in the pipeline.

## 1. Initializing Git & First Commit
I have initialized a Git repository in your `mlops-ci-cd-project` directory and committed all the files.

**What was done:**
```bash
git init
git add .
git commit -m "Initial commit: End-to-End MLOps Pipeline for Diabetes Prediction"
```

## 2. Source Code Change Demonstration
To trigger the automated GitHub Actions pipeline via a code change, we updated the hyperparameters of the Random Forest model.

**What was changed in `src/train.py`:**
- **From:** `model = RandomForestClassifier(n_estimators=100, random_state=42)`
- **To:** `model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)`

**The Commit:**
```bash
git add src/train.py
git commit -m "feat(model): Update Random Forest hyperparameters to demonstrate CI/CD source code change trigger"
```

**What the pipeline does:** When pushed to GitHub, the `.github/workflows/ml_pipeline.yml` detects a change in `src/train.py`. It boots up, installs requirements, re-runs `train.py` (which creates a new `model.pkl` with the new hyperparameters), and finally runs `pytest`.

## 3. Dataset Change Demonstration
To trigger the pipeline via a data change, we modified the first patient record in our `diabetes.csv` dataset.

**What was changed in `data/diabetes.csv`:**
- **From:** `6,148,72,35,0,33.6,0.627,50,1`
- **To:** `10,200,90,40,100,45.0,1.2,60,1` (Simulating a severe diabetes case)

**The Commit:**
```bash
git add data/diabetes.csv
git commit -m "fix(data): Update first row of dataset to demonstrate CI/CD dataset change trigger"
```

**What the pipeline does:** The workflow file detects a change in `data/diabetes.csv`. The pipeline spins up again, reading the newly changed CSV file, training the model on this updated data, saving the new `model.pkl`, and passing the tests.

---

## 🚀 How to link this to your GitHub Repository

Right now, these commits exist **locally** on your computer. To actually trigger the GitHub Actions, you need to push this local repository to a new repository on your GitHub account.

1. Go to [GitHub.com](https://github.com/) and create a **New Repository** (name it something like `mlops-diabetes-pipeline`). Do **not** initialize it with a README or .gitignore (leave it completely empty).
2. Copy the URL of your new repository.
3. Open your terminal in the `mlops-ci-cd-project` folder and run:
   ```bash
   git branch -M main
   git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
   git push -u origin main
   ```
4. Once you push, go to the **"Actions"** tab on your GitHub repository. You will see the pipeline automatically start running!
