# CryptoAI Trading Bot

## Overview

CryptoAI is a comprehensive cryptocurrency trading bot that leverages machine learning models and sentiment analysis to execute trades automatically. It integrates with multiple services and uses the ELK Stack for centralized logging and monitoring.

## Features

- **Automated Trading:** Executes buy and sell orders based on ML predictions.
- **Machine Learning Models:** Utilizes Gradient Boosting and LSTM models for price prediction.
- **Sentiment Analysis:** Analyzes market sentiment to inform trading decisions.
- **Risk Management:** Implements risk controls like maximum daily trades and loss limits.
- **Centralized Logging:** Uses ELK Stack (Elasticsearch, Logstash, Kibana) for log aggregation and visualization.
- **Notifications:** Sends alerts via Telegram and Email for key events.

## Prerequisites

- **Python 3.8 or higher**
- **Docker Desktop** (for running ELK Stack and other services)
- **Virtual Environment** (`venv` recommended)
- **API Keys** for exchanges (e.g., KuCoin, MEXC)
- **Telegram Bot Token** and **Chat ID**
- **Email Credentials** for notifications

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CryptoAI.git
cd CryptoAI
