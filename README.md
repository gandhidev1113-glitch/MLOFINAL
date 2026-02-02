# MLOps Final Project - Titanic Survival Prediction

## 📋 Project Description

This project implements an end-to-end ML Ops pipeline for predicting passenger survival on the Titanic. The project follows ML Ops best practices including environment management, code organization, preprocessing, model training, and will eventually include model serving, containerization, and monitoring.

## 🎯 Task Definition

**Problem**: Binary classification task to predict whether a passenger survived the Titanic disaster based on various features such as age, gender, class, fare, etc.

**Objective**: Build a machine learning model that can accurately predict passenger survival and deploy it as a production-ready service following ML Ops principles.

**Evaluation**: Model performance will be evaluated using accuracy, precision, recall, and F1-score metrics.

## 📊 Dataset Source

**Dataset**: [Titanic - Machine Learning from Disaster](https://www.kaggle.com/c/titanic)

**Source**: Kaggle Competition Dataset

**Description**: The dataset contains information about 891 passengers in the training set and 418 passengers in the test set. Features include:
- Passenger demographics (Age, Sex)
- Ticket information (Class, Fare, Cabin, Embarked)
- Family information (SibSp, Parch)
- Target variable: Survival (0 = No, 1 = Yes)

**Data Download**: The dataset can be automatically downloaded using the provided `download_data.py` script, which uses the Kaggle API.

## 👥 Team Member Roles

**Team Members**:
- **PAUL MICKY D COSTA** - ML Engineer / Project Setup Lead
  - Project infrastructure setup and environment management
  - Data pipeline development (download, preprocessing)
  - Baseline model training and evaluation
  - Code organization and documentation
- **DEVKUMAR PARIKSHIT GANDHI** - DevOps & Automation Engineer
  - CI/CD pipeline setup (GitHub Actions), Dockerization, and environment consistency.
- **Thai Bao DUONG** - Serving & Monitoring Engineer (FastAPI Owner)
   - Build and maintain the FastAPI inference service: POST /predict, GET /health, GET /ready with Pydantic schemas and consistent error handling.
	- Implement model loading interface (artifact path / MODEL_URI) aligned with training outputs and MLflow conventions.
	- Add basic observability: structured logs + simple runtime metrics (latency, request/error counts) suitable for Docker runtime.
	- Write API contract tests (pytest/TestClient) for /predict, /health, /ready to prevent regressions.
	- Backup support for unit testing: contribute additional unit tests when needed (especially around serving-related utilities/interfaces) and assist the Quality Lead in maintaining coverage targets.
- **Sofyen Fenich** - ML Scientist & Model Validation 
   - Implement and compare multiple models.
	- Evaluate models using suitable metrics.
	- Model Validation & Data Leakage Prevention
	- Model Explainability & Analysis
	- Keep the best model
   
- Member 4: [Name] - [Role/Responsibilities]

## 🚀 Getting Started

### Prerequisites

- Python 3.11+
- UV package manager
- Kaggle API credentials (for data download)

### Setup

1. **Install dependencies**:
   ```bash
   uv sync
   ```

2. **Set up Kaggle API** (for data download):
   - Go to https://www.kaggle.com/settings
   - Create API token and place `kaggle.json` in `~/.kaggle/` (or `C:\Users\<username>\.kaggle\` on Windows)

3. **Run the pipeline** (using main entry point): 

   Run commands from the project root (where `pyproject.toml` is located)

   ```bash
   # Download dataset
   uv run python -m src.main download

   # Preprocess data
   uv run python -m src.main preprocess

   # Train baseline model
   uv run python -m src.main train

   # Or run everything at once
   uv run python -m src.main all
   ```

   Alternatively, you can run scripts directly:
   ```bash
   # Download dataset
   uv run python -m src.download_data

   # Preprocess data
   uv run python -m src.preprocessing

   # Train baseline model
   uv run python -m src.train
   ```

## 📁 Project Structure

```
MLOFINAL/
├── data/
│   ├── raw/              # Raw dataset files
│   ├── derived/           # Preprocessed data
│   └── output/           # Generated outputs (reports, etc.)
├── src/
│   ├── __init__.py
│   ├── main.py            # Main entry point with CLI
│   ├── download_data.py   # Kaggle API data download script
│   ├── preprocessing.py   # Data preprocessing pipeline
│   ├── train.py           # Model training script
│   ├── utils.py           # Shared utility functions
│   ├── generate_plots.py  # Visualization scripts
├── models/                # Saved models (created during training)
├── pyproject.toml         # Project dependencies
├── uv.lock                # Locked dependencies
├── .gitignore             # Git ignore rules
└── README.md              # This file
```

## 🔄 Workflow

1. **Data Download**: Use `uv run python -m src.main download` (or `uv run python -m src.download_data`) to fetch the dataset from Kaggle  
2. **Preprocessing**: Run `uv run python -m src.main preprocess` (or `uv run python -m src.preprocessing`) to clean and engineer features  
3. **Training**: Execute `uv run python -m src.main train` (or `uv run python -m src.train`) to train the baseline model  
4. **Evaluation**: Model metrics are displayed during training  

### Quick Start (Full Pipeline)

```bash
uv run python -m src.main all
```

This will run: download → preprocess → train in sequence.

## 📝 Notes

- This project is part of the ML Ops course final project
- Checkpoint 1 focuses on project setup, data preprocessing, and baseline model training
- Future checkpoints will add experiment tracking (MLflow), model serving (FastAPI), containerization (Docker), and monitoring
