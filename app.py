
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import plotly.express as px

st.set_page_config(page_title="ABC Company Dashboard", layout="wide")

# --- Data Loading Functions ---
@st.cache_data
def load_kpi_data(path="ABC_Company_KPI_Data.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Revenue_Change"] = df["Revenue"].pct_change().round(4)
    df["Profit_Margin_Change"] = df["Profit_Margin"].pct_change().round(4)
    df["Inventory_Change"] = df["Inventory_Levels"].pct_change().round(4)
    return df

@st.cache_data
def load_external_events(path="ABC_Company_External_Events.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.rename(columns={"Event_Type": "Type", "Event_Title": "Message"})
    df["Source"] = "External"
    return df[["Date", "Source", "Type", "Message", "Inventory_Impact", "Revenue_Impact"]]

# --- Main App ---
def main():
    st.title("ABC Company: KPI Forecasting & Event Impact Dashboard")

    # Load data
    kpi_df = load_kpi_data()
    ext_df = load_external_events()

    # Panel 1: Event Alerts
    st.header("1. Event Alerts")
    # Internal alerts
    alerts = []
    for _, row in kpi_df.iterrows():
        date = row["Date"]
        if row["Revenue_Change"] < -0.10:
            alerts.append({"Date": date, "Source": "Internal", "Type": "Black Swan", "Message": "Revenue dropped sharply"})
        elif row["Revenue_Change"] > 0.10:
            alerts.append({"Date": date, "Source": "Internal", "Type": "White Swan", "Message": "Revenue surged significantly"})
        if row["Profit_Margin_Change"] < -0.05:
            alerts.append({"Date": date, "Source": "Internal", "Type": "Black Swan", "Message": "Profit margin declined heavily"})
        elif row["Profit_Margin_Change"] > 0.05:
            alerts.append({"Date": date, "Source": "Internal", "Type": "White Swan", "Message": "Profit margin improved significantly"})
        if row["Inventory_Change"] > 0.15:
            alerts.append({"Date": date, "Source": "Internal", "Type": "Black Swan", "Message": "Inventory spiked unexpectedly"})
        elif row["Inventory_Change"] < -0.15:
            alerts.append({"Date": date, "Source": "Internal", "Type": "White Swan", "Message": "Inventory reduced efficiently"})
    int_df = pd.DataFrame(alerts)
    combined = pd.concat([int_df, ext_df[["Date","Source","Type","Message"]]], ignore_index=True)
    combined = combined.sort_values("Date").reset_index(drop=True)
    st.dataframe(combined)

    # Panel 2: KPI Monitoring
    st.header("2. KPI Monitoring")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Revenue")
        st.line_chart(kpi_df.set_index("Date")["Revenue"])
    with cols[1]:
        st.subheader("Profit Margin")
        st.line_chart(kpi_df.set_index("Date")["Profit_Margin"])
    cols2 = st.columns(2)
    with cols2[0]:
        st.subheader("ROA")
        st.line_chart(kpi_df.set_index("Date")["ROA"])
    with cols2[1]:
        st.subheader("Inventory Levels")
        st.line_chart(kpi_df.set_index("Date")["Inventory_Levels"])

    # Panel 3: Forecasting (Revenue)
    st.header("3. Forecasting")
    df = kpi_df.copy().dropna()
    df["Month_Index"] = np.arange(len(df))
    X = df[["Month_Index"]]
    y = df["Revenue"]
    model = LinearRegression().fit(X, y)
    future_idx = np.arange(len(df), len(df)+6).reshape(-1,1)
    preds = model.predict(future_idx)
    # display
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df["Date"], y, label="Historical")
    future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.offsets.MonthBegin(),
                                 periods=6, freq="MS")
    ax.plot(future_dates, preds, linestyle="--", label="Forecast")
    ax.legend()
    st.pyplot(fig)

    # --- Panel 4: Impact Analysis ---
    st.header("4. Impact Analysis")

    # Load external events with impacts
    impact_df = load_external_events()  # must include Inventory_Impact & Revenue_Impact

    # 4a. Grouped Bar Chart of Impacts
    fig_imp = px.bar(
    impact_df,
    x="Date",
    y=["Inventory_Impact", "Revenue_Impact"],
    labels={
        "value": "Impact Amount (USD)",
        "variable": "Impact Type"
    },
    title="External Event Impacts on Inventory & Revenue"
    )
    fig_imp.update_layout(barmode="group", xaxis_tickformat="%b %Y")
    st.plotly_chart(fig_imp, use_container_width=True)

    # 4b. Annotate Forecast Chart with Event Markers
    st.subheader("Forecast with Event Annotations")
    # Assume you stored your forecast figure as `fig` above
    for _, event in impact_df.iterrows():
    fig.add_vline(
        x=event["Date"],
        line_width=1,
        line_dash="dash",
        annotation_text=event["Message"],
        annotation_position="top left",
        annotation_font_size=10
    )
    st.pyplot(fig)

if __name__ == "__main__":
    main()
