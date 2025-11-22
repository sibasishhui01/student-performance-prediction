🎓 Student Performance Prediction — ML + Flask Web App

This project is an end-to-end Machine Learning web application that predicts a student’s academic performance (Low, Average, or High) based on multiple factors such as gender, parental education, lunch type, test preparation, and exam scores.
It features a trained ML model, a Flask backend, and a modern premium UI built with HTML/CSS (Glassmorphism Style).

🚀 Features
✅ Machine Learning

Synthetic dataset of 1000 students generated programmatically

ML preprocessing pipeline using:

OneHotEncoder

StandardScaler

Models trained:

Logistic Regression

Random Forest

Gradient Boosting

Best model selected using GridSearchCV

Saved using pickle as best_student_model.pkl

✅ Backend (Flask)

Python Flask server to handle form data

Loads the trained .pkl model

Performs real-time predictions

Returns results dynamically to the frontend

✅ Frontend (UI)

Responsive modern design

Sidebar navigation

Glassmorphism theme

Neon glow buttons

Styled dropdowns and number inputs

Color-coded prediction result:

🔴 Low

🟡 Average

🟢 High

📁 Project Structure
studentpredict/
│
├── model/
│   └── best_student_model.pkl
│
└── app/
    ├── app.py
    ├── templates/
    │     └── index.html
    └── static/
          └── style.css

🧠 Tech Stack
🔹 Machine Learning

Python

scikit-learn

numpy

pandas

matplotlib

pickle

🔹 Backend

Flask

🔹 Frontend

HTML5

CSS3 (Premium UI / Glassmorphism)

🛠 How It Works

User fills the form with student details

Data is sent to the Flask backend via POST

Backend loads the trained ML model

Model predicts:

Low

Average

High

Result is displayed with a beautifully styled color card

📌 Author

Sibasish
Computer Science & Engineering (CSE)
