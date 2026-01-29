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
- Member 2: [Name] - [Role/Responsibilities]
- Member 3: [Name] - [Role/Responsibilities]
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
   ```bash
   # Download dataset
   python main.py download
   
   # Preprocess data
   python main.py preprocess
   
   # Train baseline model
   python main.py train
   
   # Or run everything at once
   python main.py all
   ```

   Alternatively, you can run scripts directly:
   ```bash
   python download_data.py
   python src/preprocessing.py
   python src/train.py
   ```

## 📁 Project Structure

```
MLOFINAL/
├── data/
│   ├── raw/              # Raw dataset files
│   ├── derived/           # Preprocessed data
│   └── output/           # Generated outputs (reports, etc.)
├── src/
│   ├── preprocessing.py  # Data preprocessing pipeline
│   ├── train.py          # Model training script
│   ├── utils.py          # Shared utility functions
│   ├── generate_plots.py # Visualization scripts
│   └── report.qmd        # Report template
├── models/                # Saved models (created during training)
├── download_data.py       # Kaggle API data download script
├── main.py               # Main entry point with CLI
├── pyproject.toml        # Project dependencies
├── uv.lock               # Locked dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🔄 Workflow

1. **Data Download**: Use `python main.py download` or `download_data.py` to fetch the dataset from Kaggle
2. **Preprocessing**: Run `python main.py preprocess` or `src/preprocessing.py` to clean and engineer features
3. **Training**: Execute `python main.py train` or `src/train.py` to train the baseline model
4. **Evaluation**: Model metrics are displayed during training

### Quick Start (Full Pipeline)
```bash
python main.py all
```

This will run: download → preprocess → train in sequence.

## 📝 Notes

- This project is part of the ML Ops course final project
- Checkpoint 1 focuses on project setup, data preprocessing, and baseline model training
- Future checkpoints will add experiment tracking (MLflow), model serving (FastAPI), containerization (Docker), and monitoring
