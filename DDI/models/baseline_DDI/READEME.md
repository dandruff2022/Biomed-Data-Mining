# How to Run the Code

## Step 1: Navigate to Your Project Directory
cd path/to/your/project

## Step 2: Specify the --data_path Argument
The --data_path argument is required.
It should point to the folder containing the dataset files.
This folder must include:
- train.json → training data
- valid.json → validation data
- test.json → test data

## Step 3: Run the Models

### Logistic Regression
python main_logistic.py --data_path data

### Support Vector Machine (SVM)
python main_svm.py --data_path data