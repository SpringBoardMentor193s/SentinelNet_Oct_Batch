███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗███╗   ██╗███████╗████████╗
██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝████╗  ██║██╔════╝╚══██╔══╝
███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██╔██╗ ██║█████╗     ██║   
╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║╚██╗██║██╔══╝     ██║   
███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗██║ ╚████║███████╗   ██║   
╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝   

          SENTINELNET • NEXT-GEN AI CYBER DEFENSE PLATFORM

<p align="center">
  <img src="https://img.shields.io/badge/STATUS-ACTIVE-success?style=for-the-badge&logo=github"/>
  <img src="https://img.shields.io/badge/STREAMLIT-SAAS%20APP-red?style=for-the-badge&logo=streamlit"/>
  <img src="https://img.shields.io/badge/AI%20MODEL-TRAINED-blue?style=for-the-badge&logo=scikitlearn"/>
  <img src="https://img.shields.io/badge/CYBERSECURITY-NIDS-black?style=for-the-badge&logo=hackthebox"/>
</p>



---

🚨 Project Overview

SentinelNet is a fully functional Network Intrusion Detection System (NIDS) built using Machine Learning, Data Mining, and Streamlit SaaS-style UI.

It detects:

🛑 DDoS attacks

🛑 Probe scans

🛑 U2R / R2L attacks

🛑 Brute force attempts

🟢 Normal traffic


The system provides:

Real-time classification

SOC-grade dashboards

Threat analytics

PDF reporting

Deep-learning ready pipeline



---

⚙️ Features

✔ Fully Automated ML Pipeline

Preprocessing

Label encoding

Feature engineering

Training & testing

Evaluation


✔ SOC-Themed Dashboard (Dark Cyber UI)

Animated neon title section

Metrics counters

Pie charts

Gauge chart

Probability graph

Trend & histogram charts


✔ Full Streamlit App

Main application file:

/workspaces/SentinelNet_Oct_Batch/streamlit_app.py

✔ Multi-Model Support

Random Forest

Gradient Boosting

Logistic Regression

SVM (RBF)

XGBoost (optional)


✔ Advanced Evaluation

Confusion matrix

ROC curve

F1-score

Precision / Recall

Classification reports


✔ PDF Report Generator

Exports metrics + plots in a professional PDF.


---

📁 Project Structure (Matches Your Repository)

SENTINELNET/
├── .venv/                     # Virtual environment
├── confusion.ipynb           # Confusion matrix experiments
├── EDA.ipynb                 # Data exploration
├── evaluation_Metrics.ipynb  # Testing models
├── fselection.ipynb          # Feature selection
├── kdd_testing.csv           # NSL-KDD test dataset
├── kdd_training.csv          # NSL-KDD train dataset
├── LICENSE
├── main.ipynb                # Main notebook for code testing
├── practice.py               # Temp script
├── preprocessing.ipynb       # Preprocessing experiments
├── README.md                 # <— replaced with this new file
├── sampledata.ipynb
├── sentinelnet_model.joblib  # Saved ML model
├── streamlit_app.py          # 💠 FINAL STREAMLIT DASHBOARD APP


---

🧠 Machine Learning Pipeline

1️⃣ Data Input

Accepts:

NSL-KDD CSV

CICIDS-2017 CSV

Any custom CSV/TXT/XLSX


2️⃣ Preprocessing

Label mapping (Normal → 0, Attack → 1)

Encoding categorical fields

Standard scaling

Feature selection

Train-test split


3️⃣ Model Training

Uses modular trainer:

RandomForestClassifier
GradientBoostingClassifier
SVC (RBF)
LogisticRegression
XGBoost (if installed)

4️⃣ Evaluation

Includes:

Accuracy

Precision

Recall

F1 score

Confusion matrix

ROC (AUC)


5️⃣ Prediction

Live predictions & abnormal probability trends.


---

📊 Dashboards & Visualizations

SentinelNet provides:

Label distribution bar graph

Feature trend line

Feature histogram

Threat distribution pie chart

SOC-style long cards

Probability time-series

Live threat meter



---

📝 PDF Report Generation

Exports:

Model used

Accuracy / Precision / Recall / F1

CM + ROC images

Threat statistics

Classification report

Timestamp

Your name & institution


One-click export.


---

🚀 Running the App

1. Activate environment

source .venv/bin/activate

2. Install dependencies

pip install -r requirements.txt

3. Launch Streamlit

streamlit run streamlit_app.py


---

📂 Datasets Used

✔ NSL-KDD

kdd_training.csv
kdd_testing.csv

✔ CICIDS-2017

Supported when converted into CSV.

✔ Custom Datasets

Any CSV with a label column works.


---

📈 Results (Example)

Model	Accuracy	Notes

Random Forest	⭐ 92–96%	Best stability
Gradient Boosting	⭐ 89–94%	Good consistency
SVM (RBF)	⭐ 90–95%	Strong for boundaries
Logistic Regression	80–85%	Baseline
XGBoost	⭐⭐ 95–99%	Top tier (optional)



---

🔐 Threat Levels

Level	Range	Meaning

🟢 Low	<10%	Safe
🟡 Medium	10–30%	Suspicious
🟠 High	30–60%	Possible attack
🔴 Critical	>60%	Attack in progress



---

🧭 Future Enhancements

LSTM-based intrusion model

Autoencoder anomaly detection

Real-time packet capture (dpkt / scapy)

SIEM integration

Cloud dashboard

Threat-intel feed

Online learning



---

🤝 Contributing

Pull requests welcome.


---

📜 License

MIT License (see LICENSE)


---

⭐ If SentinelNet helped you, please star the repository!



---

⚙️ Features

✔ Fully Automated ML Pipeline

Preprocessing

Label encoding

Feature engineering

Training & testing

Evaluation



---

✔ SOC-Themed Dashboard (Dark Cyber UI)

Animated neon title section

Metrics counters

Pie charts

Gauge chart

Probability graph

Trend & histogram charts



---

✔ Full Streamlit App

Main application file:
/workspaces/SentinelNet_Oct_Batch/streamlit_app.py


---

✔ Multi-Model Support

Random Forest

Gradient Boosting

Logistic Regression

SVM (RBF)

XGBoost (optional)



---

✔ Advanced Evaluation

Confusion matrix

ROC curve

F1-score

Precision / Recall

Classification reports



---

✔ PDF Report Generator

Exports:

Model used

Accuracy / Precision / Recall / F1

Confusion matrix + ROC images

Threat statistics

Classification report

Timestamp & user info

One-click export



---

📁 Project Structure

SENTINELNET/
├── .venv/                     # Virtual environment
├── confusion.ipynb             # Confusion matrix experiments
├── EDA.ipynb                   # Data exploration
├── evaluation_Metrics.ipynb    # Testing models
├── fselection.ipynb            # Feature selection
├── kdd_testing.csv             # NSL-KDD test dataset
├── kdd_training.csv            # NSL-KDD train dataset
├── LICENSE
├── main.ipynb                  # Main notebook for code testing
├── practice.py                 # Temp script
├── preprocessing.ipynb         # Preprocessing experiments
├── sampledata.ipynb
├── sentinelnet_model.joblib    # Saved ML model
├── streamlit_app.py            # 💠 FINAL STREAMLIT DASHBOARD APP


---

🧠 Machine Learning Pipeline

1️⃣ Data Input

Accepts:

NSL-KDD CSV

CICIDS-2017 CSV

Any custom CSV/TXT/XLSX



---

2️⃣ Preprocessing

Label mapping (Normal → 0, Attack → 1)

Encoding categorical fields

Standard scaling

Feature selection

Train-test split



---

3️⃣ Model Training

Uses modular trainer:

RandomForestClassifier

GradientBoostingClassifier

SVC (RBF)

LogisticRegression

XGBoost (if installed)



---

4️⃣ Evaluation

Includes:

Accuracy

Precision

Recall

F1 score

Confusion matrix

ROC (AUC)



---

5️⃣ Prediction

Live predictions & abnormal probability trends



---

📊 Dashboards & Visualizations

Label distribution bar graph

Feature trend line

Feature histogram

Threat distribution pie chart

SOC-style long cards

Probability time-series

Live threat meter



---

📝 PDF Report Generation

Exports:

Model used

Accuracy / Precision / Recall / F1

Confusion matrix + ROC

Threat statistics

Classification report

Timestamp, user info

One-click export



---

🚀 Running the App

# Activate environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Streamlit
streamlit run streamlit_app.py


---

📂 Datasets Used

✔ NSL-KDD
kdd_training.csv / kdd_testing.csv

✔ CICIDS-2017
Supported when converted into CSV

✔ Custom Datasets
Any CSV with a label column works


---

📈 Model Results (Example)

Model	Accuracy	Notes

Random Forest	⭐ 92–96%	Best stability
Gradient Boosting	⭐ 89–94%	Good consistency
SVM (RBF)	⭐ 90–95%	Strong for boundaries
Logistic Regression	80–85%	Baseline
XGBoost (optional)	⭐⭐ 95–99%	Top tier



---

🔐 Threat Levels

Level	Range	Meaning

🟢 Low	<10%	Safe
🟡 Medium	10–30%	Suspicious
🟠 High	30–60%	Possible attack
🔴 Critical	>60%	Attack in progress



---

🧭 Future Enhancements

LSTM-based intrusion model

Autoencoder anomaly detection

Real-time packet capture (dpkt / scapy)

SIEM integration

Cloud dashboard

Threat-intel feed

Online learning



---

🤝 Contributing

Pull requests welcome! ⭐
