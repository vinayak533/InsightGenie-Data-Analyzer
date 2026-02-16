# 🚀 InsightGenie — AI-Powered Data Intelligence Platform

## 📌 Summary

InsightGenie is an AI-powered data analysis platform that converts raw CSV datasets into actionable insights through automated data profiling, visualization, and intelligent machine learning recommendations. It enables users to understand datasets quickly without writing code using a FastAPI backend and React frontend.

---

## 🛠️ Technologies Used

### Backend

* Python
* FastAPI
* Pandas
* Scikit-learn

### Frontend

* React
* Vite
* Tailwind CSS
* JavaScript

### Other

* REST APIs
* Data Visualization Libraries
* Modular Architecture

---

## ✨ Features

* Automated data quality analysis and dataset health score
* Missing value and duplicate detection
* Interactive visualizations (histograms, scatter plots, heatmaps)
* Intelligent ML problem type detection (classification, regression, clustering)
* Algorithm recommendations with explanations
* Target variable detection and class imbalance analysis
* Large dataset handling with optimized performance
* Modern responsive UI with smooth animations

---

## ⌨️ Keyboard Shortcuts

```
Ctrl + C   → Stop backend server
Ctrl + C   → Stop frontend server
Enter      → Execute command
```

---

## ⚙️ Process

```
1. User uploads CSV dataset
2. Backend performs automated data profiling
3. Data quality metrics and statistics are generated
4. Visualizations are created
5. ML advisor detects problem type and suggests algorithms
6. Results are displayed in the frontend dashboard
```

---

## 🏗️ How I Built It

```
- Designed system architecture with FastAPI backend and React frontend
- Implemented automated EDA pipeline using Pandas
- Created ML advisor module using Scikit-learn logic
- Developed REST APIs for communication between frontend and backend
- Built interactive UI using React, Vite, and Tailwind CSS
- Optimized performance using sampling and lazy loading techniques
```

---

## 📚 What I Learned

```
- Full-stack AI application development
- FastAPI backend architecture and API design
- React frontend development and integration
- Automated exploratory data analysis techniques
- Machine learning problem detection logic
- Performance optimization for large datasets
- End-to-end project structuring
```

---

## 🚀 How It Could Be Improved

```
- Add AutoML model training and evaluation
- Cloud deployment (AWS / GCP / Azure)
- User authentication and project saving
- Exportable PDF or dashboard reports
- Real-time collaboration features
- More advanced visualization options
```

---

## ▶️ How to Run the Project

### Clone Repository

```bash
git clone https://github.com/yourusername/InsightGenie-Data-Analyzer.git
cd InsightGenie-Data-Analyzer
```

---

### Backend Setup

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

Backend runs at:

```
http://127.0.0.1:8000
```

---

### Frontend Setup

```bash
npm install
npm run dev
```

Frontend runs at:

```
http://localhost:5173
```

---

## 📂 Project Structure

```
InsightGenie-Data-Analyzer/
│
├── main.py
├── eda.py
├── ml_advisor.py
├── schemas.py
├── requirements.txt
│
├── frontend/
│   ├── index.html
│   ├── index.js
│   ├── package.json
│   └── vite.config.js
│
└── README.md
```

---

## ⭐ About

InsightGenie is an AI-powered platform that automatically analyzes CSV datasets, evaluates data quality, generates visualizations, and recommends suitable machine learning models with clear explanations.
