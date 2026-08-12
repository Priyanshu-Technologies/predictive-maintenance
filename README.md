# Predictive Maintenance

<p align="center">
  <strong>End-to-end machine failure prediction using XGBoost, FastAPI, Docker, and cloud deployment.</strong>
</p>

<p align="center">
  <a href="https://predictive-maintenance-ldma.onrender.com/">
    <img src="https://img.shields.io/badge/Live%20Demo-Online-success?style=for-the-badge" alt="Live Demo">
  </a>
  <a href="https://github.com/Priyanshu-Technologies/predictive-maintenance">
    <img src="https://img.shields.io/github/stars/Priyanshu-Technologies/predictive-maintenance?style=for-the-badge" alt="GitHub Stars">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.14-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/XGBoost-3.4.0-FF6600?style=flat-square&logo=xgboost&logoColor=white" alt="XGBoost">
  <img src="https://img.shields.io/badge/scikit--learn-1.9.0-F7931E?style=flat-square&logo=scikit-learn&logoColor=white" alt="Scikit-learn">
  <img src="https://img.shields.io/badge/FastAPI-0.115.6-009688?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Docker-Containerized-2496ED?style=flat-square&logo=docker&logoColor=white" alt="Docker">
  <img src="https://img.shields.io/badge/Render-Deployed-46E3B7?style=flat-square&logo=render&logoColor=white" alt="Render">
</p>

---

## 🚀 Overview

Predictive Maintenance is an end-to-end machine learning application designed to estimate the probability of machine failure from sensor and operating-condition data.

The project takes a trained XGBoost model and turns it into a deployable application through:

- Exploratory Data Analysis (EDA)
- Data preprocessing
- Feature engineering
- Machine learning model training
- Model evaluation
- Model serialization
- FastAPI REST API
- Interactive web interface
- Docker containerization
- Cloud deployment

The application allows users to enter a machine's current operating conditions and receive an estimated failure probability and corresponding risk level.


### 🌐 Live Demo

**[Open the Live Application](https://predictive-maintenance-ldma.onrender.com/)**

> The application is deployed on Render's free tier. The service may spin down after a period of inactivity, so the first request after inactivity may take longer than subsequent requests.

---


## ✨ Features

- 🧠 XGBoost-based machine failure prediction
- 📊 Sensor-based predictive maintenance inference
- 🔧 Feature engineering and preprocessing
- 🎯 Failure probability estimation
- 🚦 Risk-level classification
- ⚡ FastAPI REST API
- 🖥️ Interactive browser-based interface
- 🐳 Dockerized application
- ☁️ Cloud deployment
- 📚 Automatic FastAPI Swagger documentation
- 🔁 Reproducible inference across local, Docker, and cloud environments

---


## 🏗️ System Architecture

```text
                    ┌─────────────────────┐
                    │    User / Browser   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Web Interface    │
                    │     HTML / CSS / JS  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │    REST API Layer   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Feature Processing │
                    │   + Preprocessing   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │     XGBoost Model   │
                    │   + Model Artifacts │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Failure Probability │
                    │    + Risk Level     │
                    └─────────────────────┘

👨‍💻 Author
Priyanshu Sharma
