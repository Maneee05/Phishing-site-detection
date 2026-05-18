🛡️ Phishing Website Detection using Machine Learning

A web application that detects whether a given website URL is phishing or safe, built using Python, scikit-learn, and Streamlit.
This project uses a Random Forest Classifier trained on a phishing dataset to analyze URL-based features and make predictions.

🚀 Features

🧠 Machine Learning model trained using Random Forest

🌐 Interactive Streamlit web app – enter a website URL and get instant results

⚡ Real-time prediction without accessing or visiting the website

📊 Clean, user-friendly UI

💾 Model stored and loaded using joblib

🧩 Technologies Used

Python 3.x

Scikit-learn

Pandas

NumPy

Streamlit

Joblib

Matplotlib / Seaborn (for EDA)

⚙️ How to Run Locally

1. Clone the Repository
   
   git clone https://github.com/Maneee05/Phishing-site-detection.git
   
   cd Phishing-site-detection

2. Install Dependencies
   
   pip install -r requirements.txt

3. Run the App
   
   streamlit run app.py

Streamlit will open automatically at:
👉 http://localhost:8501

🧠 Model Details

Algorithm: Random Forest Classifier

Dataset: Phishing Websites Dataset (Kaggle)

Features: URL-based characteristics (length, special characters, subdomains, etc.)

Libraries Used: scikit-learn, pandas, numpy

The trained model is saved as:

joblib.dump(rf_model, "phishing_rf_model.pkl")

And loaded in app.py as:

model = joblib.load("phishing_rf_model.pkl")

👩‍💻 Author

Maneesha Manohar

B.Tech CSE Student | Aspiring ML & Cybersecurity Engineer

🔗 LinkedIn - www.linkedin.com/in/maneesha-manohar-607819249

🔗 GitHub - https://github.com/Maneee05
