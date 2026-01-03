# Domain-Aware Hierarchical Stacked Ensemble (DAHSE) for Postpartum Depression

**A machine learning framework for the robust and accurate prediction of Postpartum Depression (PPD) using demographic and health indicators.**

## Project Overview
Postpartum Depression (PPD) is a critical public health concern. This project implements **DAHSE**, a **Domain-Aware Hierarchical Stacked Ensemble** framework, to predict PPD risk levels in mothers.

By stacking multiple base classifiers (Random Forest, XGBoost, LightGBM) and optimizing them with a meta-learner, this model addresses challenges like class imbalance and non-linear feature interactions common in medical datasets.

## Methodological Contribution
The core contribution of this work is a hierarchical ensemble architecture:
1.  **Level-0 (Base Learners):** A diverse set of strong classifiers captures different patterns in the data.
2.  **Level-1 (Meta-Learner):** A Logistic Regression meta-learner combines these predictions to reduce variance and improve generalization.
3.  **Domain-Aware Preprocessing:** Custom cleaning pipelines handle categorical variables (e.g., medical history, demographics) and missing values specific to the dataset context.

## Repository Structure
* `dahse-for-ppd.ipynb`: Main Jupyter Notebook containing:
    * **Data Loading & Cleaning:** Handling of `PPD_dataset_v2.csv`.
    * **EDA:** Exploratory Data Analysis of maternal health factors.
    * **Model Training:** Implementation of the Stacked Ensemble.
    * **Evaluation:** Performance metrics (Accuracy, F1-Score, ROC-AUC).
* `PPD_dataset_v2.csv`: The dataset used for training and testing.
* `README.md`: Project documentation.

## How to Run
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/arnabsroy9/DAHSE-for-PPD.git](https://github.com/arnabsroy9/DAHSE-for-PPD.git)
    cd DAHSE-for-PPD
    ```
2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost
    ```
3.  **Launch the Notebook:**
    ```bash
    jupyter notebook dahse-for-ppd.ipynb
    ```
4.  **Execute:** Run all cells. Ensure `PPD_dataset_v2.csv` is in the same directory as the notebook.

## Dataset & Citation
This project utilizes the **"Data for Postpartum Depression Prediction in Bangladesh"** dataset.

* **Source:** Mendeley Data
* **Filename:** `PPD_dataset_v2.csv`
* **Description:** Contains sociodemographic, familial, and medical history data from 800+ participants.
* **Citation:**
    > Raisa, Jasiya Fairiz; Kaiser, M Shamim (2025), "Data for Postpartum Depression Prediction in Bangladesh", Mendeley Data, V2, doi: 10.17632/4nznnrk8cg.2

## Results
The DAHSE model demonstrates a strong ability to identify high-risk cases (high sensitivity), which is the primary clinical objective:

* **Accuracy:** 73.8% (on hold-out test set)
* **Sensitivity (Recall):** 85.2% — *Crucial for medical screening.*
* **ROC-AUC:** 0.77
* **Key Insight:** While overall accuracy is balanced, the hierarchical ensemble significantly minimizes **False Negatives** (achieving high recall), ensuring that mothers at risk of PPD are rarely missed by the screening tool.
