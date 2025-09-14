# CattleCare AI - AI-Powered Farm Insights

CattleCare AI is a web-based application that leverages AI and ML models to provide **cattle and farm management insights**. Farmers can predict milk yield, monitor cattle health, and receive guidance using an interactive chatbot. The project integrates **Flask**, **Tailwind CSS**, and **Google Gemini API** for intelligent responses.

---

## 🐄 Features

* **Animal Prediction**

  * Predict milk yield using regression models.
  * Predict health risks and disease probabilities using classification models.
  * Store animal data with timestamps in a database.
  * Visualize top 10 recent animals in styled cards with risk alerts.

* **Chatbot (CattleCare AI)**

  * Interactive AI chatbot located at the **bottom-right corner** of the page.
  * Answers only questions related to **cattle, cows, dairy, and farm management**.
  * Powered by **Google Gemini API** for intelligent responses.
  * User-friendly interface with input box and send button.
  * Real-time display of conversation.

* **Dashboard**

  * Responsive UI using Tailwind CSS.
  * Cards highlight **high-risk animals** with red alerts.
  * Shows predicted milk yield, health status, age, weight, breed, lactation stage, and parity.

* **Data Management**

  * Add, store, and visualize animal data.
  * Export reports in CSV or PDF format.
  * Supports farm-level data analytics.

---

## 📦 Requirements

* Python 3.11+
* Flask
* Flask-CORS
* Flask-Login
* Flask-Mail (optional, for email notifications)
* Pandas
* Numpy
* Joblib
* Requests
* Dotenv
* Tailwind CSS (for front-end)
* Google Gemini API key

Install Python dependencies using:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
Flask
Flask-Cors
Flask-Login
Flask-Mail
pandas
numpy
joblib
requests
python-dotenv
```

---

## ⚙️ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/CattleCare-AI.git
cd CattleCare-AI
```

2. **Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Set environment variables**

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_google_gemini_api_key
```

5. **Run the Flask app**

```bash
python app.py
```

6. **Open the app in browser**

Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000)

* Chatbot will appear at the **bottom-right corner**.
* Add new animal records via input forms.
* View predictions and risk cards.

---

## 🛠 Project Structure

```
CattleCare-AI/
├── app.py                        # Main Flask app
├── auth.py                       # Authentication logic
├── test.py                       # Testing APIs / chatbot functionality
│
├── templates/                    # Frontend HTML templates
│   └── index.html
│
├── static/                       # Frontend assets
│   ├── css/                      # Tailwind CSS / styles
│   ├── js/                       # JavaScript files
│   ├── img/                      # Images, icons
│   └── mp3/                      # Audio files (alerts/notifications)
│
├── models/                       # Trained ML models
│   ├── classifier.pkl
│   ├── classifier_joblib.pkl
│   ├── label_encoder.pkl
│   ├── labelencoder_joblib.pkl
│   ├── preprocessor.pkl
│   ├── preprocessor_joblib.pkl
│   ├── regressor.pkl
│   ├── regressor_joblib.pkl
│   ├── anomaly_model.pkl
│   └── mappings.pkl
│
├── data/                         # Datasets
│   ├── cattle_synthetic.csv
│   ├── login_data.csv
│   └── login_with_anomalies.csv
│
├── instance/                     # App/database instance
│   └── app.db
│
├── database/                     # (optional extra DB location if scaling)
│   └── users.db
│
├── scripts/                      # Utility scripts
│   ├── data_generator.py         # Generate cattle data
│   ├── synthetic_data.py         # Synthetic dataset generator
│   ├── simulate_stream.py        # Streaming simulation
│   ├── features.py               # Feature extraction
│   └── train_model.py            # Train ML models
│
├── deployment/                   # Deployment & configs
│   ├── Dockerfile
│   ├── docker-compose.yml
│   └── .env
│
├── tests/                        # Unit & integration tests
│   └── test_api.py
│
├── requirements.txt              # Python dependencies
├── README.md                     # Documentation
├── LICENSE                       # License file
└── .gitignore                    # Git ignore rules
```

---

## 💡 Notes

* Only messages related to **cattle, cows, farm, dairy, livestock** are processed by the chatbot.
* Ensure **Google Gemini API key** is valid.
* Tailwind CSS is loaded via CDN for styling.
* Recent animal cards display risk visually using **red and green backgrounds**.
* The chatbot interface supports **Enter key** and **Send button**.

---

## 📧 Contact

For questions, feedback, or contributions:

* Developer: Aman Nayak
* Email: `your-email@example.com`
* GitHub: [https://github.com/yourusername](https://github.com/yourusername)

---

## ✅ License

This project is open-source and available under the **MIT License**.
