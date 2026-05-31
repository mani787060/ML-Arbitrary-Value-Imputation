# Data Imputation Architectures: Arbitrary Value & Statistical Feature Recovery
[![Machine Learning](https://img.shields.io/badge/Domain-Data%20Engineering-blue)](https://scikit-learn.org/)
[![Preprocessing](https://img.shields.io/badge/Strategy-Arbitrary%20%26%20Statistical%20Imputation-orange)](https://scikit-learn.org/stable/modules/impute.html)
[![Dataset](https://img.shields.io/badge/Dataset-Titanic%20Toy%20Slices-green)](./titanic_toy.csv)

## 🏗️ Project Overview
Real-world machine learning workflows frequently encounter missing features that cannot simply be dropped via listwise deletion without discarding valuable data volume. When data is **Missing Not At Random (MNAR)**—meaning the missingness carries a specific hidden signal—standard statistical defaults like mean or median imputation can smooth out the data too much, accidentally hiding that signal from downstream classification or regression models.

This repository explores **Arbitrary Value Imputation**, an engineering technique where missing entries are filled with a pre-defined, extreme constant outside the feature's natural mathematical range (e.g., `-1`, `999`, or `-999`). Using the **Titanic Toy Dataset** (`titanic_toy.csv`), this project runs a comprehensive comparative benchmark evaluating how arbitrary constant assignment shifts feature distributions, distorts variances, and affects model inputs compared to standard **Mean** and **Median** imputation methods.

---

## 🛠️ Advanced Engineering Mechanics

### 1. The Statistical Assumptions & Behavioral Signals
* **Capturing MNAR Patterns:** Arbitrary value imputation is particularly powerful when the missingness itself is predictive. For example, if a missing `Age` or `Cabin` value on the Titanic indicates that the data was never recorded due to emergency scenarios, filling it with `-999` preserves that flag. This allows tree-based algorithms to split on that exact boundary, capturing the underlying behavioral pattern.
* **The Variance Shifting Risk:** Because both statistical averages and arbitrary constants fill missing points with an absolute fixed value, they introduce an artificial spike in the data distribution. This spike can distort the standard deviation, alter covariance structures, and compress the feature's natural range.

### 2. The Operational Flow
The ingestion pipeline scans independent sparse columns (such as `Age` or `Fare`), splits the data stream across competing imputation methods simultaneously, and maps the resulting shifts in variance and distribution shapes.


```text
                 ┌─────────────────────────────────────┐
                 │  Sparse Titanic Feature Vector      │
                 └─────────────────────────────────────┘
                                   │
                                   ▼

                 ┌─────────────────────────────────────┐
                 │      Missing Value Strategies       │
                 └─────────────────────────────────────┘
                                   │
        ┌──────────────────────────┼──────────────────────────┐
        │                          │                          │
        ▼                          ▼                          ▼

 Mean / Median              Constant = -1           Constant = -999
  Imputation              Missingness Indicator      Extreme Marker
        │                          │                          │
        ▼                          ▼                          ▼

 Reduced Variance       Preserve Missing Signal    Tree-Based Models
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼

                 ┌─────────────────────────────────────┐
                 │     Imputed Feature Matrix          │
                 └─────────────────────────────────────┘
```


---

## 🔬 Implementation Workflows

The source notebook `ML-Arbitrary-Value-Imputation.ipynb` executes a highly structured experimentation matrix:

1. **Missing Data Quantifying:** Running Pandas profiling checks to evaluate missingness rates across continuous parameters like `Age` and `Fare`.
2. **Statistical Imputation Baselines:** Utilizing Scikit-Learn's `SimpleImputer(strategy='mean')` and `SimpleImputer(strategy='median')` to compute standard data-filling baselines.
3. **Arbitrary Value Configuration:** Implementing extreme constant strategies using both custom Pandas maps and `SimpleImputer(strategy='constant', fill_value=-1)` or `fill_value=-999`.
4. **Distribution Shift Auditing:** Generating overlayed Seaborn KDE distribution curves and calculating variance differences ($\Delta \sigma^2$) **Before vs. After** imputation to track exactly how the data spread changes.
5. **Downstream Model Input Validation:** Assessing how these different values impact model inputs, showing how linear models react to massive arbitrary outliers compared to how tree models use them.

---

## 📊 Imputation Strategy Comparison Matrix

| Imputation Modality | Mathematical Injection Profile | Impact on Overall Variance | Best Downstream Estimator |
| :--- | :--- | :--- | :--- |
| **Mean Imputation** | Fills with dataset center ($\mu$) | Severely underestimates variance; creates a massive central peak | Linear / Distance-Based Models |
| **Median Imputation**| Fills with the 50th percentile | Stabilizes variance better than Mean against skewed outliers | Distance-Based Models / Neural Networks |
| **Arbitrary Constant** | Places extreme flags outside range ($-\infty$ or $+\infty$) | Creates severe variance distortions and artificial outliers | Decision Trees / Random Forests / Gradient Boosters |

---

## 💻 Tech Stack & Requirements
* **Language Environment:** Python 3.9+
* **Data Handling Infrastructure:** Pandas, NumPy
* **Core Preprocessing Pipelines:** Scikit-Learn (`impute.SimpleImputer`, `model_selection`)
* **Statistical Visualization Engine:** Matplotlib, Seaborn
* **Execution Workspace:** Jupyter Notebook Runtime

---

## 🚀 Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/your-username/ML-Arbitrary-Value-Imputation.git](https://github.com/your-username/ML-Arbitrary-Value-Imputation.git)
    cd ML-Arbitrary-Value-Imputation
    ```
2.  **Install Essential Dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn jupyter
    ```
3.  **Execute the Diagnostics Pipeline:**
    ```bash
    jupyter notebook
    ```
    Open `ML-Arbitrary-Value-Imputation.ipynb` to step through the comparative evaluations and inspect the distribution variance charts.
