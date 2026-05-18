# 🛡️ Phishing Website Detection using Machine Learning

An end-to-end Machine Learning-powered phishing website detection system built using Python and Streamlit that analyzes URLs and predicts whether a website is legitimate or phishing.

This project combines Cybersecurity and Machine Learning to detect malicious websites using extracted URL-based features and a trained ML classification model.

---

## 🚀 Features

- 🔍 Real-time phishing URL prediction
- 🤖 Machine Learning-based classification
- 🌐 Streamlit interactive web interface
- 📊 URL feature extraction pipeline
- ⚡ Fast and lightweight prediction system
- 🛡️ Cybersecurity-focused practical application
- 📈 User-friendly dashboard for testing URLs

---

## 🧠 Problem Statement

Phishing attacks are one of the most common cyber threats where attackers create fake websites to steal sensitive information such as:

- Login credentials
- Banking information
- Personal data

This project aims to automatically detect phishing websites using Machine Learning techniques based on URL characteristics and engineered security features.

---

## 🏗️ Project Architecture
```text
User URL Input
       ↓
Feature Extraction
       ↓
Preprocessing
       ↓
Trained ML Model
       ↓
Prediction Output
       ↓
Legitimate / Phishing
```

---

## 🛠️ Tech Stack
### Programming Language
Python
### Libraries & Frameworks
Streamlit
Scikit-learn
Pandas
NumPy
Pickle
### Machine Learning
Supervised Classification
Feature Engineering
URL-based Security Analysis

---

## 📂 Project Structure
```text
Phishing-site-detection/
│
├── phishing-streamlit/
│   ├── plots/
│   │   └── evaluation.png
│   │
│   ├── app.py
│   ├── feature_names.json
│   ├── phishing_dataset.csv
│   ├── phishing_model.pkl
│   ├── requirements.txt
│   └── train_model.py
│
├── .gitignore
└── README.md
```

---

## ⚙️ How to Run Locally

1. Clone the Repository
   
   git clone https://github.com/Maneee05/Phishing-site-detection.git
   
2. Move into the project directory:

   cd Phishing-site-detection/phishing-streamlit

3. Install Dependencies
   
   pip install -r requirements.txt

4. Run the App
   
   streamlit run app.py

---

## 💻 Usage

Launch the Streamlit application

Enter a website URL

Click on Predict

The system classifies the URL as:

Legitimate Website

Phishing Website

---

## 📸 Application Preview

# <img width="1903" height="775" alt="Screenshot 2026-05-18 222226" src="https://github.com/user-attachments/assets/9af6941d-b670-4eb2-ae4d-b055c205c08a" />

# <img width="1247" height="431" alt="Screenshot 2026-05-18 222303" src="https://github.com/user-attachments/assets/a9e9477d-c3d5-47fc-8e6f-227eea561641" />

# <img width="1916" height="726" alt="Screenshot 2026-05-18 222408" src="https://github.com/user-attachments/assets/6171ca75-9782-49b5-b6d7-c106429f0dbf" />

---

## 📈 Future Improvements
Deep Learning-based detection

Real-time domain reputation lookup

Browser extension integration

API deployment using FastAPI

Explainable AI (XAI) visualizations

Threat intelligence integration

---

## 👩‍💻 Author
Maneesha Manohar

Computer Science undergraduate passionate about:

AI/ML

Applied Machine Learning

Data Science

🔗 LinkedIn - www.linkedin.com/in/maneesha-manohar-607819249

🔗 GitHub - https://github.com/Maneee05

---

⭐ If you found this project useful, give it a star!
