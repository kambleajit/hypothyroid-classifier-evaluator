# Hypothyroid Screening Classifiers

## Problem statement

Build and compare five binary classification models that identify hypothyroid cases from patient screening records. All hypothyroid diagnosis labels are grouped into the positive class; records labeled `negative` form the negative class. Recall is the primary metric because about 8% of records are hypothyroid: a false negative may delay follow-up and treatment, while a false positive leads to a confirmatory blood test. Therefore, missed positive cases are treated as more costly than additional false alarms.

## Dataset description

The project uses the UCI Thyroid Disease `allhypo` dataset, downloading `allhypo.data` and `allhypo.test` from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/). It contains **3,772 samples** and **30 model input features** after preprocessing and encoding. The source contains patient demographics, clinical indicators, thyroid laboratory measurements, referral source, and diagnosis. Preprocessing in `train_models.py` treats `?` values as missing, removes TBG columns, median-imputes numeric values, imputes implausible ages above 120, encodes binary flags and sex, one-hot encodes referral source, and standardizes the features. The data is split into stratified 80% training and 20% test partitions with random state 42.

The app accepts labeled CSV files with the source feature columns and a `class` target column. The included `test_data.csv` contains the held-out test partition. Uploaded rows are transformed using the saved training preprocessing and are never used to retrain a model.

## Code Repository Link

GitHub repository URL: https://github.com/kambleajit/hypothyroid-classifier-evaluator.git

## Models used

Logistic Regression, Decision Tree Classifier, K-Nearest Neighbors (5 neighbors), Gaussian Naive Bayes, and a sigmoid-calibrated Support Vector Machine. Logistic Regression, Decision Tree, and SVM use balanced class weights to address the imbalanced screening labels. All models use the same stratified split and standardized feature matrix.

### Evaluation comparison

Metrics below are calculated on the held-out test split. `AUC` uses predicted positive-class probabilities.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.9735 | 0.9837 | 0.7568 | 0.9655 | 0.8485 | 0.8416 |
| Decision Tree | 0.9974 | 0.9907 | 0.9828 | 0.9828 | 0.9828 | 0.9813 |
| KNN | 0.9523 | 0.8090 | 0.9583 | 0.3966 | 0.5610 | 0.5998 |
| Naive Bayes | 0.2040 | 0.6223 | 0.0868 | 0.9828 | 0.1594 | 0.0966 |
| Support Vector Machine | 0.9709 | 0.9625 | 0.9737 | 0.6379 | 0.7708 | 0.7753 |

### Model observations

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | High recall (0.9655) and AUC (0.9837), with lower precision (0.7568); balanced weights catch most positive cases but include more false positives. |
| Decision Tree | Best overall holdout results: 0.9974 accuracy, 0.9907 AUC, 0.9828 recall, and 0.9813 MCC. It balances minority-class detection and precision particularly well on this split. |
| KNN | Excellent precision (0.9583) but recall is only 0.3966, so it misses many hypothyroid cases at the default decision threshold. |
| Naive Bayes | Recall is high (0.9828), but precision (0.0868) and accuracy (0.2040) are poor, showing that its predictions produce many false positives on this dataset. |
| Support Vector Machine | The calibrated, balanced SVM has high precision (0.9737), but recall is 0.6379; it misses more positive cases than Logistic Regression and the Decision Tree. |

**Overall, winner for this dataset:** Decision Tree, which achieved the strongest accuracy, AUC, precision, F1, and MCC on the fixed stratified holdout while retaining 0.9828 recall. Treat this as a single-split comparison rather than evidence of clinical performance.

## Run locally

```powershell
python -m pip install -r requirements.txt
python train_models.py
python -m streamlit run app.py
```

The training script downloads the public UCI files, trains and evaluates all five models, then writes model artifacts to `model/`, metrics to `model/metrics.json`, and the labeled held-out CSV to `test_data.csv`. The Streamlit app loads only these saved artifacts when evaluating uploads.