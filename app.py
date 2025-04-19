
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.express as px

st.set_page_config(page_title="ABC Company Dashboard", layout="wide")

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
    df = df.rename(columns={"Event_Type": "Type", "Event_Title": "Message"})
    df["Source"] = "External"
    return df[["Date", "Source", "Type", "Message", "Inventory_Impact", "Revenue_Impact"]]

# --- Model Training ---
@st.cache_resource
def train_revenue_model(X, y):
    return LinearRegression().fit(X, y)

# --- Helper Display Functions ---
def show_kpi_summary(kpi_df):
    latest = kpi_df.iloc[-1]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Latest Revenue", f"${latest['Revenue']:,.0f}", f"{latest['Revenue_Change']*100:.1f}%")
    c2.metric("Latest Profit Margin", f"{latest['Profit_Margin']:.2%}", f"{latest['Profit_Margin_Change']*100:.1f}%")
    c3.metric("Latest ROA", f"{latest['ROA']:.2%}", f"{latest['ROA_Change']*100:.1f}%")
    c4.metric("Latest Inventory", f"{latest['Inventory_Levels']:,.0f}", f"{latest['Inventory_Change']*100:.1f}%")

def show_event_alerts(kpi_df, ext_df):
    st.header("1. Event Alerts")
    alerts = pd.concat([
        kpi_df.loc[kpi_df['Revenue_Change'] < -0.10, ['Date']]
            .assign(Source='Internal', Type='Black Swan', Message='Revenue dropped sharply'),
        kpi_df.loc[kpi_df['Revenue_Change'] >  0.10, ['Date']]
            .assign(Source='Internal', Type='White Swan', Message='Revenue surged significantly'),
        kpi_df.loc[kpi_df['Profit_Margin_Change'] < -0.05, ['Date']]
            .assign(Source='Internal', Type='Black Swan', Message='Profit margin declined heavily'),
        kpi_df.loc[kpi_df['Profit_Margin_Change'] >  0.05, ['Date']]
            .assign(Source='Internal', Type='White Swan', Message='Profit margin improved significantly'),
        kpi_df.loc[kpi_df['Inventory_Change'] >  0.15, ['Date']]
            .assign(Source='Internal', Type='Black Swan', Message='Inventory spiked unexpectedly'),
        kpi_df.loc[kpi_df['Inventory_Change'] < -0.15, ['Date']]
            .assign(Source='Internal', Type='White Swan', Message='Inventory reduced efficiently')
    ], ignore_index=True)
    combined = pd.concat([alerts, ext_df[['Date','Source','Type','Message']]], ignore_index=True)
    combined = combined.sort_values('Date').reset_index(drop=True)
    st.dataframe(combined)

def show_kpi_monitoring(kpi_df):
    st.header("2. KPI Monitoring")
    cols = st.columns(2)
    with cols[0]:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(kpi_df['Date'], kpi_df['Revenue'], marker='o')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b ’%y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel("Revenue")
        ax.set_title("Revenue Trend")
        st.pyplot(fig)
    with cols[1]:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(kpi_df['Date'], kpi_df['Profit_Margin'], marker='o')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b ’%y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel("Profit Margin")
        ax.set_title("Profit Margin Trend")
        st.pyplot(fig)
    cols2 = st.columns(2)
    with cols2[0]:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(kpi_df['Date'], kpi_df['ROA'], marker='o')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b ’%y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel("ROA")
        ax.set_title("ROA Trend")
        st.pyplot(fig)
    with cols2[1]:
        fig, ax = plt.subplots(figsize=(6,3))
        ax.plot(kpi_df['Date'], kpi_df['Inventory_Levels'], marker='o')
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b ’%y"))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_ylabel("Inventory Levels")
        ax.set_title("Inventory Levels Trend")
        st.pyplot(fig)

def show_forecasting(kpi_df, kpi_col, horizon):
    st.header("3. Forecasting")
    df_f = kpi_df.dropna().copy()
    df_f['Month_Index'] = np.arange(len(df_f))
    X, y = df_f[[ 'Month_Index' ]], df_f[kpi_col]
    model = train_revenue_model(X, y)
    future_idx = np.arange(len(df_f), len(df_f)+horizon).reshape(-1,1)
    preds = model.predict(future_idx)
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(df_f['Date'], y, label='Historical')
    future_dates = pd.date_range(start=df_f['Date'].iloc[-1] + pd.offsets.MonthBegin(), periods=horizon, freq='MS')
    ax.plot(future_dates, preds, linestyle='--', label=f'Forecast ({kpi_col})')
    ax.legend()
    st.pyplot(fig)
    return fig

def show_impact_analysis(ext_df, forecast_fig):
    st.header("4. Impact Analysis")
    imp_fig = px.bar(ext_df, x='Date', y=['Inventory_Impact','Revenue_Impact'], barmode='group',
                     labels={'value':'Impact (USD)','variable':'Type'}, title='External Event Impacts')
    st.plotly_chart(imp_fig, use_container_width=True)
    st.subheader("Forecast with Event Annotations")
    ax = forecast_fig.axes[0]
    for _, ev in ext_df.iterrows():
        ax.axvline(x=ev['Date'], color='red', linestyle='--')
        ax.text(ev['Date'], ax.get_ylim()[1], ev['Message'], rotation=45, ha='right', va='bottom', fontsize=8)
    st.pyplot(forecast_fig)

def main():
    kpi_df = load_kpi_data()
    ext_df = load_external_events()

    # Sidebar filters
    st.sidebar.header('Controls')
    event_types = ext_df['Type'].unique().tolist()
    selected_event_types = st.sidebar.multiselect('Event Types', event_types, default=event_types)
    kpi_options = ['Revenue', 'ROA']
    selected_kpi = st.sidebar.selectbox('Select KPI for Forecast', kpi_options, index=0)
    horizon_options = [3, 6, 12]
    selected_horizon = st.sidebar.selectbox('Forecast Horizon (months)', horizon_options, index=1)

    # Apply filters
    ext_df = ext_df[ext_df['Type'].isin(selected_event_types)]

    st.title("ABC Company: KPI Dashboard")
    show_kpi_summary(kpi_df)
    show_event_alerts(kpi_df, ext_df)
    show_kpi_monitoring(kpi_df)
    forecast_fig = show_forecasting(kpi_df, selected_kpi, selected_horizon)
    show_impact_analysis(ext_df, forecast_fig)

if __name__ == "__main__":
    main()
