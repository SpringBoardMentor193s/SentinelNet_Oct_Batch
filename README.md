# 🚨 SentinelNet – AI-Powered Network Intrusion Detection System (NIDS)

**SentinelNet** is an AI-powered Network Intrusion Detection System (NIDS) capable of identifying malicious network traffic and cyber-attacks in real time.  
Using advanced Machine Learning techniques, it classifies traffic as **Normal, DoS, DDoS, PortScan, Botnet, Web Attack, and more**.

This project is designed for **students, researchers, and cybersecurity beginners** who want to understand how ML can be used to detect intrusions.

---

## 🔥 Key Features

- ✅ Real-time packet sniffing using **Scapy**
- ✅ Preprocessing pipelines for **NSL-KDD** & **CICIDS2017**
- ✅ ML models:  
  - Logistic Regression  
  - Random Forest  
  - Gradient Boosting  
  - XGBoost  
- ✅ Exportable trained model (`.pkl`)
- ✅ Feature extraction & dataset balancing
- ✅ Beautiful **Streamlit dashboard** with live detection
- ✅ Auto-generated **PDF report** of predictions

---

## 📁 Project Structure

NIDS_Project/
│── src/
│ ├── train_cicids.py
│ ├── train_nslkdd.py
│ ├── preprocessing_cicids.py
│ ├── preprocessing_nslkdd.py
│ ├── realtime_sniffer.py
│ ├── evaluate.py
│ └── utils.py
│
│── data_nslkdd/
│ ├── KDDTrain+.txt
│ └── KDDTest+.txt
│
│── assets/
│── artifacts/
│── report.pdf
│── app.py (Streamlit app)
│── requirements.txt
│── README.md

📊 Model Training Workflow

Dataset → Preprocessing → Feature Engineering → Model Training → Evaluation → Deployment

Architecture Diagram:

[ Raw Network Traffic ]
↓
[Preprocessing]
↓
[Feature Extraction]
↓
[ML Model]
↓
[Prediction: Normal / Attack]

## 📚 Datasets Used

We used two industry-standard security datasets:

### **1️⃣ NSL-KDD**
- Cleaned KDD dataset
- Balanced realistic attack distribution  
- Used for classical ML evaluation  

Dataset link (Google Drive):  
👉 (https://drive.google.com/drive/folders/1syIrfBupLrgAHISp9V7nZKMzd-rVg5z5?usp=sharing)

---

### **2️⃣ CICIDS 2017**
- Real, modern intrusion dataset  
- Includes: DDoS, PortScan, Brute Force, Botnet, Infiltration, Web Attacks  
- Used for advanced model training

Dataset link (Google Drive):  
👉 https://drive.google.com/drive/folders/1VPCmPJfjnoBPYWJoD3SV-U9r5Tg50wAL?usp=sharing

**Note:** Large datasets are not stored inside the GitHub repo.

---

## 🖥️ Streamlit App (Live Intrusion Detection)

Run the dashboard:

streamlit run app.py
Features:

🔴 Live network monitoring

📊 Attack probability predictions

📄 Auto-generated PDF Reports

🌐 Interactive UI

🧪 Model Evaluation
Model	Accuracy	Precision	Recall	F1 Score
Logistic Regression	84.7%	82%	83%	82%
Random Forest	96.1%	95%	96%	96%
Gradient Boosting	94.3%	93%	94%	94%
XGBoost	97.5%	97%	98%	97%

(Replace these with real values later)

📦 Installation
Clone the project:
git clone https://github.com/SpringBoardMentor193s/SentinelNet_Oct_Batch.git
cd NIDS_Project
pip install -r requirements.txt

🧠 Skills Demonstrated
Machine Learning

Data Preprocessing

Cybersecurity Concepts

Python Automation

Network Packet Sniffing

Streamlit UI Development

Git & Version Control

💡 Future Improvements
Deep Learning-based intrusion detection

Auto-updating threat signatures

Distributed NIDS architecture

Docker-based deployment

🤝 Contributing
Pull requests are welcome! For major updates, please open an issue first to discuss.

📜 License
This project is licensed under the MIT License.

❤️ Developer
Aryan Sahu
AI & Cybersecurity Enthusiast
