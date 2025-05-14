
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px
import google.generativeai as genai

# --- Page Configuration ---
st.set_page_config(page_title="ABC Company Dashboard", layout="wide")

# --- Gemini API Configuration ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])

# --- Data Loading Functions ---
@st.cache_data
def load_kpi_data(path="ABC_Company_KPI_Data.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Revenue_Change"] = df["Revenue"].pct_change().round(4)
    df["Profit_Margin_Change"] = df["Profit_Margin"].pct_change().round(4)
    df["ROA_Change"] = df["ROA"].pct_change().round(4)
    df["Inventory_Change"] = df["Inventory_Levels"].pct_change().round(4)
    return df

@st.cache_data
def load_external_events(path="ABC_Company_External_Events.csv"):
    df = pd.read_csv(path, parse_dates=["Date"])
    df["Source"] = "External"
    return df

# --- Helper Functions ---
def show_kpi_summary(df):
    latest = df.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Latest Revenue", f"${latest['Revenue']:,.0f}", f"{latest['Revenue_Change']*100:.1f}%")
    col2.metric("Latest Profit Margin", f"{latest['Profit_Margin']:.2%}", f"{latest['Profit_Margin_Change']*100:.1f}%")
    col3.metric("Latest ROA", f"{latest['ROA']:.2%}", f"{latest['ROA_Change']*100:.1f}%")
    col4.metric("Latest Inventory Level", f"${latest['Inventory_Levels']:,.0f}", f"{latest['Inventory_Change']*100:.1f}%")

def show_selected_event_alerts(ext_df):
    st.header("1. External Event Alerts")
    
    available_types = ext_df["Type"].unique().tolist()
    selected_types = st.multiselect("Filter by Event Type:", options=available_types, default=available_types)

    filtered_events = ext_df[ext_df["Type"].isin(selected_types)]
    prioritized_events = filtered_events.sort_values(by="Revenue_Impact", key=abs, ascending=False).head(6)

    for _, event in prioritized_events.iterrows():
        with st.container():
            st.subheader(f"{event['Type']}: {event['Message']}")
            st.write(f"Date: {event['Date'].strftime('%b %d, %Y')}")
            st.write(f"Revenue Impact: ${event['Revenue_Impact']:,.0f}")
            st.write(f"Inventory Impact: ${event['Inventory_Impact']:,.0f}")
            st.link_button("View Details", event['Details_URL'])
            st.markdown("---")

def show_kpi_monitoring(df):
    st.header("2. KPI Monitoring")
    cols = st.columns(2)
    with cols[0]:
        st.subheader("Revenue")
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["Revenue"], marker='o')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with cols[1]:
        st.subheader("Profit Margin")
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["Profit_Margin"], marker='o')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))
        plt.xticks(rotation=45)
        st.pyplot(fig)
    cols2 = st.columns(2)
    with cols2[0]:
        st.subheader("ROA")
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["ROA"], marker='o')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))
        plt.xticks(rotation=45)
        st.pyplot(fig)
    with cols2[1]:
        st.subheader("Inventory Levels")
        fig, ax = plt.subplots()
        ax.plot(df["Date"], df["Inventory_Levels"], marker='o')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))
        plt.xticks(rotation=45)
        st.pyplot(fig)

def show_forecasting(df, selected_kpi="Revenue", horizon=6):
    st.header("3. Forecasting")
    df_forecast = df.dropna()
    df_forecast["Month_Index"] = np.arange(len(df_forecast))
    X = df_forecast[["Month_Index"]]
    y = df_forecast[selected_kpi]
    model = LinearRegression().fit(X, y)
    future_idx = np.arange(len(df_forecast), len(df_forecast) + horizon).reshape(-1, 1)
    preds = model.predict(future_idx)

    fig, ax = plt.subplots()
    ax.plot(df_forecast["Date"], y, label="Historical")
    future_dates = pd.date_range(start=df_forecast["Date"].iloc[-1] + pd.offsets.MonthBegin(), periods=horizon, freq="MS")
    ax.plot(future_dates, preds, linestyle="--", label="Forecast")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b\n%y'))
    plt.xticks(rotation=45)
    st.pyplot(fig)

    return fig

def show_impact_analysis(ext_df, forecast_fig):
    st.header("4. Impact Analysis")
    fig_imp = px.bar(
        ext_df,
        x="Date",
        y=["Inventory_Impact", "Revenue_Impact"],
        labels={"value": "Impact Amount (USD)", "variable": "Impact Type"},
        title="External Event Impacts on Inventory & Revenue"
    )
    fig_imp.update_layout(barmode="group", xaxis_tickformat="%b %Y")
    st.plotly_chart(fig_imp, use_container_width=True)

    st.subheader("Forecast with Event Annotations")
    for _, ev in ext_df.iterrows():
        forecast_fig.gca().axvline(x=ev["Date"], color="red", linestyle="--")
        forecast_fig.gca().annotate(
            ev["Message"],
            xy=(ev["Date"], 0),
            xytext=(ev["Date"], 0.05),
            rotation=45,
            textcoords="offset points",
            ha='right'
        )
    st.pyplot(forecast_fig)

def show_chatbot():
    if "chat_open" not in st.session_state:
        st.session_state.chat_open = False

    chat_cols = st.columns([8, 1])
    with chat_cols[1]:
        if st.button("💬 Open Chat", key="chat_button"):
            st.session_state.chat_open = not st.session_state.chat_open

    if st.session_state.chat_open:
        st.subheader("KPI Assistant")
        user_query = st.text_input("Type your question here:")
        if user_query:
            with st.spinner('Thinking...'):
               try:
                    model = genai.GenerativeModel("gemini-pro")
                    response = model.generate_content(f"Question: {user_query}")
                    st.success(response.text)
                    except Exception as e:
                    st.error("Chatbot is currently unavailable or quota is exhausted. Please try again later."

def main():
    kpi_df = load_kpi_data()
    ext_df = load_external_events()

    st.title("ABC Company: KPI Dashboard")

    show_chatbot()
    show_kpi_summary(kpi_df)
    show_selected_event_alerts(ext_df)
    show_kpi_monitoring(kpi_df)
    forecast_fig = show_forecasting(kpi_df)
    show_impact_analysis(ext_df, forecast_fig)

if __name__ == "__main__":
    main()
