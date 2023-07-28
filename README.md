# Anomaly Detection in Large Datasets: A  Case Study in Loan Defaults

Given the rise in loan defaults, especially after the COVID-19 pandemic, it is necessary to predict if customers might default on a loan for risk management. This thesis proposes an early warning system architecture using anomaly detection based on the unbalanced nature of loan default data in the real world. Most customers do not default on their loans; only a tiny percentage do, resulting in an unbalanced dataset. We aim to evaluate potential anomaly detection methods for their suitability in handling unbalanced datasets. We conduct a comparative study on different anomaly detection approaches on four balanced and unbalanced datasets.

We compare five of each supervised, unsupervised, and semi-supervised anomaly detection approaches. The supervised algorithms compared are logistic regression, stochastic gradient descent (SGD), XGBoost, LightGBM, and CatBoost classification methods. The unsupervised anomaly detection methods are isolation forest, angle-based outlier detection (ABOD), outlier detection using empirical cumulative distribution function (ECOD), copula-based outlier detection (COPOD), and deep one-class classifier with autoencoder (DeepSVDD). The semi-supervised anomaly detection methods are improving supervised outlier detection with unsupervised representation learning (XGBOD), feature encoding with autoencoders for weakly-supervised anomaly detection (FeaWAD), deep semi-supervised anomaly detection (DeepSAD), progressive image deraining networks (PReNet), and deep anomaly detection with deviation networks (DevNet).

We compare them using standard evaluation metrics such as accuracy, precision, recall, F1 score, training and prediction time, and area under the receiver operating characteristic (ROC) curve. The results show that anomaly detection methods perform significantly better on unbalanced loan default data and are more suitable for real-world applications. The results also show that supervised methods work better for balanced datasets, and for peer-to-peer lending datasets, boosting approaches are expected to perform well.


## Scripts
Four independent scripts have been created for experimentation on each dataset as described below.

|File name                |Script                          |Dataset|Data file name|
|---------------|----------------------------------|--------------------------------|--------------------------------|
|`bondora_script.py`|Experimentation on the Bondora dataset|[Bondora Peer-to-Peer Lending Data (IEEE DataPort)](https://ieee-dataport.org/open-access/bondora-peer-peer-lending-data)            |`Bondora_preprocessed.csv`|
|`autoloan_script.py`|Experimentation on the L&T Vehicle Loan dataset|[L&T Vehicle Loan Default Prediction Data (Kaggle)](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction)            |`train.csv`|
|`deloittemh_script.py`|Experimentation on the Deloitte-MachineHack Loan Default dataset|[Deloitte-MachineHack Loan Default Prediction Dataset (Kaggle)](https://www.kaggle.com/datasets/hemanthsai7/loandefault)            |`train.csv`|
|`lendingclub_script.py`|Experimentation on the LendingClub dataset|[Lending Club 2007-2020Q3 Dataset (Kaggle)](https://www.kaggle.com/datasets/ethon0426/lending-club-20072020q1)            |`Loan_status_2007-2020Q3.gzip`|

## Usage
1. Clone the repository. 
```bash
git clone https://github.com/RayhaanPirani/AnomalyDetectionEarlyWarningLoanDefaults.git
cd AnomalyDetectionEarlyWarningLoanDefaults
```
2. Ensure that the expected versions of the libraries used are available. The `requirements.txt` file can be used to install the expected versions.
```bash
pip install -r requirements.txt
```
3. Download the required datasets by using the above table as a reference.
4. Place the file referred by the **Data file name** column in the table in the working directory `AnomalyDetectionEarlyWarningLoanDefaults`.
5. The L&T Vehicle Loan dataset and the Deloitte-MachineHack Loan Default dataset have the same name `train.csv` in the data source. If experiments are being run for the Deloitte-MachineHack Loan Default dataset, rename the dataset's `train.csv` to `deloitte_train.csv`.
6. Run the desired script using Python. This can either be done on the command line by using the `python` command or using an IDE such as Spyder or PyCharm. It is highly recommended to use an IDE and run the code block-by-block because some models require a large amount of time to train and/or predict.
7. The average of the metrics (accuracy, precision, recall, F1 score, AUC ROC, training time, and prediction time) for each approach would be printed for the prediction evaluation and also for the evaluation of the 30% and 70% prediction probability thresholds.
