import streamlit as st
import pandas as pd
import datetime as dt

st.set_page_config(
    page_title="Clustering Analysis",
    page_icon=":material/communities:",
    layout="wide"
)

@st.cache_data(max_entries=2)
def load_clean_data():
    df = pd.read_csv('./data/cleaned/clean_retail.csv')
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["Revenue"] = df["UnitPrice"] * df["Quantity"]
    return df

st.title("Customer Clustering Dashboard")

data = load_clean_data()



agg_data = data.groupby("CustomerID", as_index=False)\
    .agg(
        LTDValue=("Revenue", "sum"),
        PurchaseFrequency = ("InvoiceNo", "nunique"),
        MostRecentPurchase = ("InvoiceDate", "max")
        )
    
max_date = pd.to_datetime(data["InvoiceDate"]).max()

agg_data["PurchaseRecency"] = (max_date - pd.to_datetime(agg_data["MostRecentPurchase"])).dt.days


st.subheader("Cleaned Data")
st.dataframe(data)

st.subheader("Grouped Data")
st.dataframe(agg_data[["CustomerID", "LTDValue", "PurchaseFrequency", "PurchaseRecency"]])

