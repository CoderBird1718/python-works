# Machine Learning Projects

A collection of four end-to-end machine learning projects covering both classification and regression tasks. Each project follows a structured pipeline — data loading, EDA, preprocessing, model training, evaluation, and comparison.

---

## Projects Overview

| Project | Type | Models Used | Target |
|---|---|---|---|
| Cancer Prediction | Classification | Logistic Regression, Random Forest | Malignant / Benign |
| Chance of Admission | Regression | Linear, Polynomial, Ridge, Lasso | Admission Probability |
| Credit Card Default Prediction | Classification | Logistic Regression, Random Forest | Default / No Default |
| Fish Weight Prediction | Regression | Linear, Polynomial, Ridge, Lasso | Fish Weight (g) |

---

## Projects

### 1. Cancer Prediction

Predicts whether a tumour is **malignant or benign** based on cell nucleus measurements from the Breast Cancer Wisconsin dataset.

**Pipeline:**
- Label encodes the `diagnosis` column (M/B → 1/0)
- Drops irrelevant columns (`id`, `Unnamed: 32`)
- Imputes missing values with column means
- Scales features with `StandardScaler`
- Trains Logistic Regression and Random Forest classifiers
- Evaluates with accuracy, classification report, confusion matrix, and ROC-AUC curve
- Plots top 10 most important features from Random Forest

**Key techniques:** `LabelEncoder`, `SimpleImputer`, `StandardScaler`, ROC curve, feature importance

---

### 2. Chance of Admission

Predicts a student's **probability of graduate school admission** (0 to 1) based on GRE score, TOEFL score, university rating, SOP, LOR, CGPA, and research experience.

**Pipeline:**
- Drops `Serial No.` column
- Plots correlation heatmap to understand feature relationships
- Trains four regression models: Linear, Polynomial (degree 2), Ridge, and Lasso
- Evaluates all models using R² score, MSE, and RMSE
- Produces an actual vs. predicted scatter plot
- Compares all models in a summary table

**Key techniques:** `PolynomialFeatures`, `Ridge`, `Lasso`, R² score, model comparison

---

### 3. Credit Card Default Prediction

Predicts whether a customer will **default on their credit card payment** using financial and demographic features.

**Pipeline:**
- Engineers a `Loan to Income` ratio feature
- Visualises class distribution and income vs. loan scatter plot
- Imputes missing values; applies stratified train-test split
- Trains Logistic Regression (baseline) and Random Forest (advanced)
- Evaluates with accuracy, classification report, confusion matrix, and ROC-AUC curve
- Plots feature importances
- Tunes Random Forest hyperparameters with `GridSearchCV` (n_estimators, max_depth)
- Prints final accuracy comparison of both models

**Key techniques:** Feature engineering, `GridSearchCV`, stratified split, ROC-AUC, hyperparameter tuning

---

### 4. Fish Weight Prediction

Predicts the **weight of a fish** (in grams) from physical measurements: length, height, width, and species.

**Pipeline:**
- One-hot encodes the `Species` column using `pd.get_dummies`
- Drops highly correlated length features (`Length1`, `Length2`)
- Plots pairplot, correlation heatmap, actual vs. predicted, and residual distribution
- Trains four models: Linear Regression, Polynomial Regression (degree 3), Ridge, and Lasso
- Compares all models by R² score and MSE in a summary DataFrame

**Key techniques:** `get_dummies`, multicollinearity removal, residual analysis, `PolynomialFeatures` (degree 3)

---

## Tech Stack

- **Python 3.x**
- [NumPy](https://numpy.org/) — numerical computing
- [Pandas](https://pandas.pydata.org/) — data manipulation
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) — visualisation
- [Scikit-learn](https://scikit-learn.org/) — machine learning models, preprocessing, and evaluation

---

## Getting Started

**1. Clone the repository**
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Install dependencies**
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

**3. Run a notebook**

Open any `.ipynb` file in Jupyter Notebook or JupyterLab:
```bash
jupyter notebook Cancer_Prediction_project.ipynb
```

All datasets are loaded directly from public URLs inside each notebook — no manual download needed.

---

## Repository Structure

```
├── Cancer_Prediction_project.ipynb
├── Chance_of_Admission_project.ipynb
├── Credit_Card_Default_Prediction.ipynb
├── Fish_Weight_Prediction.ipynb
└── README.md
```

---

## Author

Made with curiosity and scikit-learn.  
Feel free to fork, explore, and build on these projects.
