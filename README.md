# ABC Company: KPI Forecasting & Event Impact Dashboard

An interactive Streamlit app that helps business leaders:

- Monitor key performance indicators (Revenue, Profit Margin, ROA, Inventory)  
- Forecast future KPI values using a simple Linear Regression model  
- Display “Black Swan” / “White Swan” event alerts (internal anomalies + external events)  
- Analyze the quantitative impact of external events on Revenue & Inventory  

---

## 🔗 Live Demo

[Insert your Streamlit Cloud URL here]

---

## 🚀 Quick Start

1. **Clone the repo**  
   ```bash
   git clone https://github.com/<your-username>/abc-dashboard.git
   cd abc-dashboard
   ```

2. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the app**  
   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure

```
abc-dashboard/
├── app.py
├── ABC_Company_KPI_Data.csv
├── ABC_Company_External_Events.csv
├── requirements.txt
└── README.md
```

- **app.py**  
  Main Streamlit script with modular functions for each panel.

- **ABC_Company_KPI_Data.csv**  
  Dummy monthly KPI data (2023–2024).

- **ABC_Company_External_Events.csv**  
  List of external “Black Swan” & “White Swan” events with revenue & inventory impacts.

- **requirements.txt**  
  Python dependencies for Streamlit, data handling, modeling, and visualization.

---

## 🛠️ Features & Panels

1. **KPI Summary**  
   At-a-glance metrics for the latest Revenue, Profit Margin, ROA, and Inventory levels.

2. **Event Alerts**  
   Combined internal (data-driven) and external event alerts, labeled Black Swan or White Swan.

3. **KPI Monitoring**  
   Time-series line charts for core KPIs.

4. **Forecasting**  
   3-month Revenue forecast using a cached Linear Regression model; overlay on historical data.

5. **Impact Analysis**  
   Bar chart of Inventory & Revenue impacts for each external event; forecast plot annotated with event lines.

---

## 🔧 Configuration & Theming

- To switch between light/dark mode, add a `config.toml` in `.streamlit/`:
  ```toml
  [theme]
  base="dark"    # or "light"
  ```

- Adjust the forecast horizon by changing `periods` in `show_forecasting()`.

---

## 📈 Future Enhancements

- Add a sidebar filter to choose different KPIs (e.g., Forecast ROA).  
- Support longer forecast windows (6–12 months) or advanced models (ARIMA, Prophet).  
- Pull live financial data from a REST API instead of static CSV.  
- Enhance interactivity with Plotly callbacks or Altair selection widgets.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

*Prepared as part of the M.Sc. Data Science Project at SSODL — “KPI-Based Corporate Performance Forecasting Dashboard.”*
