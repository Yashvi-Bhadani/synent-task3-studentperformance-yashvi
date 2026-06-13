# Student Performance Analysis & Prediction

## Problem Statement
Analyzing what factors influence student academic performance and building a classification model to predict whether a student falls into Low, Medium, or High performance categories based on their lifestyle, family background, and academic history.

## Dataset Details
- Source: UCI Student Performance Dataset
- File: student_data.csv
- Key columns: G1 (Period 1 grade), G2 (Period 2 grade), G3 (Final grade), studytime, failures, absences, Mjob, Fjob, Walc (weekend alcohol), Dalc (daily alcohol), internet, romantic, famrel, freetime, goout, Medu, Fedu

## Approach

### 1. Exploratory Analysis
- Described dataset using .info(), .describe() for both numeric and categorical columns
- Confirmed zero null values and zero duplicates

### 2. Feature Engineering
- Created Avg Score = (G1 + G2 + G3) / 3
- Created Performance Category using pd.qcut into 3 quantile-based groups: Low, Medium, High

### 3. Aggregation & Pivot Analysis
- Pivot tables to analyze average score by Mother's Job, Father's Job, Weekend Alcohol consumption, and Family Size
- Key finding: Students whose mothers worked in health or teaching scored higher on average

### 4. Correlation Analysis
- Computed full correlation matrix
- Plotted horizontal bar chart of feature correlations with Performance Category
- G1, G2 (previous grades) and failures were the strongest correlators

### 5. Feature Selection
- Used SelectKBest with f_classif to identify top 10 features
- Final feature set: G1, G2, studytime, failures, internet, romantic, Mjob, Fjob, absences, guardian, freetime

### 6. Model Training
- Applied One-Hot Encoding for remaining categoricals
- Stratified train-test split (80/20)
- Trained Random Forest Classifier (200 estimators, max_depth=10, class_weight='balanced')
- Compared against Logistic Regression

## Results
- Random Forest achieved higher accuracy on multi-class performance prediction
- G1 and G2 (prior grades) were the strongest predictors by a wide margin
- Number of failures was the strongest negative predictor
- Weekend alcohol consumption showed a negative correlation with performance

## Tools Used
- Python, Pandas, NumPy, Scikit-learn
- Matplotlib, Seaborn
- Jupyter Notebook
