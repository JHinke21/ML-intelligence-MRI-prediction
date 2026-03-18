# Predicting Intelligence from Structural MRI Data                                                                                        

**Author:** Johanna Hinke

**Course:** 6PASNMLN - Machine Learning in Neuroscience

**Institution:** King's College London

**Academic Year:** 2025-2026

## Research Question

**Can static brain structure predict cognitive performance?**

This project explores whether structural MRI features can predict intelligence scores using machine learning techniques. Answering this question has implications for understanding the neurobiological basis of intelligence and the limits of structural biomarkers in cognitive prediction. Despite the theoretical foundation that brain structure relates to cognitive function, prediction remains challenging due to weak effect sizes, high dimensionality, and multicollinearity inherent in neuroimaging data.

## Project Overview

### Dataset:
- **Participants:** 740 healthy individuals
- **Features:** 116 MRI-derived structural brain features (cortical thickness, brain volumes, regional metrics)
- **Target Variable:** IST Intelligence Total Score
- **Data Split:** 80% training (592 participants), 20% testing (148 participants)

### Key Findings
- **Optimal Model:** ElasticNet + PCA (n_components=5, alpha=10, l1_ratio=0.7)
- **Cross-Validation Performance:** R² = 0.0236 (p = 0.0099, statistically significant)
- **Test Set Performance:** R² = 0.0822, MAE = 31 points
- **Interpretation:** Static brain structure alone explains only ~8% of intelligence variance, suggesting dynamic brain activity and environmental factors play larger roles.

## Repository Structure:
```
mln-intelligence-prediction/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore rules
├── notebooks/                        # Jupyter notebooks for exploration
│   ├── k22016090_explorationdata.ipynb              # Main exploratory analysis
│   ├── k22016090_feature_engineering_exploration.ipynb  # Feature engineering experiments
│   └── k22016090_exploratory.ipynb                  # Additional exploration
├── src/                              # Source code
│   └── k22016090_optimal.py          # Final optimal pipeline script
├── data/                             # Data directory (not tracked)
│   └── .gitkeep
└── predictions/                      # Model predictions 
    └── K22016090_predictions.csv     # Final predictions on holdout set
```

## Methodology

### 1. Data Preprocessing
- **Cleaning:** Removed participant IDs to prevent data leakage
- **Feature Selection:** Eliminated 7 zero-variance features (e.g., `Left-WM-hypointensities`, `trigger`, `scan_end`)
- **Final Feature Count:** 109 predictive features
- **Data Quality:** No missing values detected; 2 outliers identified (|z| > 3) but retained

### 2. Model Selection Strategy
Following the **No Free Lunch Theorem** (Wolpert & Macready, 1997) which states that no single algorithm works best for all problems, I systematically compared four model families using **5-fold cross-validation** with **GridSearchCV** for hyperparameter optimization:

#### Model Families Explored:
1. **Baseline Models**
- Linear Regression: Diagnostic for multicollinearity (R² = -0.27, failed as expected)

2. **Regularized Linear Models**
- **Ridge:** Handles multicollinearity via L2 regularization
- **Lasso:** Performs feature selection via L1 regularization
- **ElasticNet:** Combines L1 and L2 penalties for stability and feature selection
- **With/Without PCA:** Tested dimensionality reduction to address curse of dimensionality

3. **Non-Linear Models**
- **Support Vector Regression (SVR) with RBF kernel:** Captures complex non-monotonic relationships
- **K-Nearest Neighbors (KNN):** Instance-based learning for local patterns

4. **Ensemble Models**
- **Random Forest:** Captures complex feature interactions
- **XGBoost:** Handles weak signals through gradient boosting with regularization

### 3. Key Results
| Model              | CV R²    | CV MSE   | Best Hyperparameters |
|--------------------|----------|----------|----------------------|
| **ElasticNet+PCA** | **0.0236** | **1553.04** | alpha=10, l1_ratio=0.7, n_components=5 |
| Lasso + PCA        | 0.0235   | 1553.24  | alpha=10, n_components=15 |
| RandomForest       | 0.0218   | 1554.47  | max_depth=5, n_estimators=100 |
| ElasticNet         | 0.0210   | 1557.11  | alpha=10, l1_ratio=0.1 |
| Ridge + PCA        | 0.0184   | 1559.50  | alpha=200, n_components=5 |
| SVR (RBF)          | 0.0041   | 1585.58  | C=1, gamma=0.01, epsilon=1 |
| KNN                | -0.0348  | 1646.90  | n_neighbors=15, weights='uniform' |
| Linear Regression  | -0.2686  | 2011.72  | (no hyperparameters) |

**Observations:**
- **PCA was transformative:** Converted negative R² to positive for Ridge and Lasso
- **Non-linear models failed:** SVR and KNN performed poorly, suggesting relationships are linear or signal is too weak
- **Ensemble models:** Comparable to linear models but no improvement

### 4. Statistical Validation

**Permutation Testing** (n=100 permutations):
- **ElasticNet + PCA:** Actual R² = 0.0236, Permutation mean = -0.0125, **p = 0.0099**
- **Lasso + PCA:** Actual R² = 0.0235, Permutation mean = -0.0122, **p = 0.0099**

Both models achieved **statistically significant** predictive performance (p < 0.05), confirming a weak but real neurobiological relationship between brain structure and intelligence.

### 5. Model Selection Rationale

**ElasticNet + PCA** was selected as the optimal model for three reasons:

1. **Best Cross-Validation Performance:** Highest R² among all models tested
2. **Robustness to Multicollinearity:** Combines L1 (feature selection) and L2 (stability) regularization—critical given high intercorrelation of brain features (confirmed via correlation heatmap)
3. **Interpretability:** Linear model with PCA dimensionality reduction (5 components) provides neurobiologically interpretable results while addressing high dimensionality


## Installation & Usage

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/JHinke21/ML-intelligence-MRI-prediction
cd mln-intelligence-prediction
```

2. **Install dependencies:**
```bash
 pip install -r requirements.txt
```                                                                                                                                     

### Running the Optimal Pipeline

1. **Prepare your data:**
 - Place `data.csv` (training data) and `holdout.csv` (holdout data) in the project root directory
 - Ensure CSV files contain participant IDs and MRI features

2. **Run the optimal pipeline:**
```bash
cd src
python k22016090_optimal.py
```

3. **Output:**
- Predictions saved to `predictions.csv`
- Summary statistics printed to console

### Exploring the Analysis
Open Jupyter notebooks to explore the full analysis:
```bash
jupyter notebook notebooks/k22016090_explorationdata.ipynb                                                                               
```

**Notebooks included:**
- `k22016090_explorationdata.ipynb`: Comprehensive exploratory analysis with 10+ models tested
- `k22016090_feature_engineering_exploration.ipynb`: Feature engineering experiments (hemisphere asymmetry, domain-specific features, polynomial interactions)
- `k22016090_exploratory.ipynb`: Additional exploratory work


## Project References
1. “A Tutorial on Support Vector Regression” Alex J. Smola, Bernhard Schölkopf - Statistics and Computing archive Volume 14 Issue 3, August 2004, p. 199-222.
2. Fernández-Blázquez, M.A., Ruiz-Sánchez de León, J.M., Sanz-Blasco, R. et al. XGBoost models based on non imaging features for the prediction of mild cognitive impairment in older adults. Sci Rep 15, 29732 (2025). https://doi.org/10.1038/s41598-025-14832-0
3. Greene, A.S., Gao, S., Scheinost, D. et al. Task-induced brain state manipulation improves prediction of individual traits. Nat Commun 9, 2807 (2018). https://doi.org/10.1038/s41467-018-04920-3
4. Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning: Data Mining, Inference, and Prediction (2nd ed.). Springer.
5. Hussain, M. A., Grant, P. E., & Ou, Y. (2024). Inferring neurocognition using artificial intelligence on Brain Mris. Frontiers in Neuroimaging, 3. https://doi.org/10.3389/fnimg.2024.1455436 
6. Lewis, J. D., Imani, V., & Tohka, J. (2024). Intelligence and cortical morphometry: Caveats in brain-behavior associations. Brain Structure and Function, 229(6), 1417–1432. https://doi.org/10.1007/s00429-024-02792-6 
7. Mwangi, B., Tian, T. S., & Soares, J. C. (2013). A review of feature reduction techniques in neuroimaging. Neuroinformatics, 12(2), 229–244. https://doi.org/10.1007/s12021-013-9204-3 
8. “Neighbourhood Components Analysis”, J. Goldberger, S. Roweis, G. Hinton, R. Salakhutdinov, Advances in Neural Information Processing Systems, Vol. 17, May 2005, pp. 513-520.Sarica et al., 2017
9. Rasero, J., Sentis, A. I., Yeh, F.-C., & Verstynen, T. (2021). Integrating across neuroimaging modalities boosts prediction accuracy of cognitive ability. PLOS Computational Biology, 17(3). https://doi.org/10.1371/journal.pcbi.1008347 
10. Sarica, A., Cerasa, A., & Quattrone, A. (2017). Random Forest algorithm for the classification of neuroimaging data in alzheimer’s disease: A systematic review. Frontiers in Aging Neuroscience, 9. https://doi.org/10.3389/fnagi.2017.00329 
11. Sui, J., Jiang, R., Bustillo, J., & Calhoun, V. (2020). Neuroimaging-based individualized prediction of cognition and behavior for Mental Disorders and Health: Methods and promises. Biological Psychiatry, 88(11), 818–828. https://doi.org/10.1016/j.biopsych.2020.02.016 
12. Vieira, B. H., Pamplona, G. S., Fachinello, K., Silva, A. K., Foss, M. P., & Salmon, C. E. (2022). On the prediction of human intelligence from neuroimaging: A systematic review of methods and reporting. Intelligence, 93, 101654. https://doi.org/10.1016/j.intell.2022.101654 
13. Wolpert, D. H., & Macready, W. G. (1997). No free lunch theorems for optimization. IEEE Transactions on Evolutionary Computation, 1(1), 67–82. https://doi.org/10.1109/4235.585893 

## License
Academic project — contact author for usage permissions. 
