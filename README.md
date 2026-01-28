# gdg_induction

## ---------------------------- All three tasks are in code.py ----------------------------

# 📈 StockAI: Real-Time Financial Intelligence Platform

### 🚀 Induction Task Submission (PS1)

**StockAI** is a comprehensive equity research dashboard built to democratize financial analysis. It integrates **Real-Time Data Streaming**, **Predictive Machine Learning**, and **Generative AI (RAG)** into a single, cohesive interface.

The platform is designed to assist analysts by automating data retrieval, visualizing live market trends, and providing context-aware AI insights.

---

## 📂 Repository Structure (Submission Compliance)

This repository is structured according to the **PS1 Induction Guidelines**:

| Directory | Module | Description |
| :--- | :--- | :--- |
| `PS1/Task1/` | **Predictive Core** | Contains the **Prophet** logic for time-series forecasting and trend analysis. |
| `PS1/Task2/` | **AI Analyst** | Contains the **RAG Pipeline** (Retrieval-Augmented Generation) integrating Gemini 1.5 Flash with live news search. |
| `PS1/Task3/` | **Live Engine** | Contains the **Real-Time Polling Architecture** logic for live chart rendering and latency management. |
| `app.py` | **Main Application** | The central Streamlit entry point that unifies all three tasks into the user interface. |
| `requirements.txt` | **Dependencies** | List of all Python libraries required to run the environment. |

---

## 🌟 Key Features

### 1. 🔮 Predictive Modelling (Task 1)
Instead of simple linear regression, this module utilizes  **Facebook Prophet** to model non-linear stock trends.
- **Methodology:** Time-series split (95% Train / 5% Test) to prevent data leakage.
- **Metrics:** Evaluated using MAE (Mean Absolute Error) and RMSE to ensure robustness against volatility.
- **Output:** Generates a 7-day future price forecast with visual trend components.

### 2. 🧠 AI Financial Analyst (Task 2)
A **RAG-based Chatbot** that goes beyond static answers. It retrieves live context before generating a response.
- **Retrieval:** Fetches real-time news using `duckduckgo-search` and market metadata via `yfinance`.
- **Augmentation:** Injects live price, 52-week highs/lows, and news headlines into the prompt context.
- **Generation:** Uses **Google Gemini 1.5 Flash** to synthesize this data into actionable financial insights.
- **Guardrails:** Includes strict prompting instructions to prevent hallucination and ensure responses stay grounded in the retrieved data.

### 3. ⚡ Live Market Stream (Task 3)
A real-time data visualization engine that simulates a trading terminal.
- **Architecture:** Implements a **Polling Architecture** with `st.empty()` containers to render updates without full-page reloads.
- **Latency Management:** Optimized refresh intervals (60s) to adhere to API rate limits while maintaining data continuity.
- **Visuals:** Interactive Plotly charts that update dynamically as new price packets arrive.

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **LLM Engine:** Google Gemini 1.5 Flash
- **Data Source:** Yahoo Finance (yfinance), DuckDuckGo Search (DDGS)
- **Machine Learning:**  Scikit-Learn, Prophet
- **Visualization:** Plotly Go, Plotly Express

---


