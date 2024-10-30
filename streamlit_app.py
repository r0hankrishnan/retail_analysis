import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
from sklearn.cluster import KMeans

#########################
####### CONFIGS
#########################
st.set_page_config(
    page_title="Clustering Analysis",
    page_icon=":material/communities:",
    layout="wide"
)

#########################
####### CACHE
#########################
@st.cache_data(max_entries=2)
def load_clean_data():
    df = pd.read_csv('./data/cleaned/clean_retail.csv')
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis=1)
    
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["Revenue"] = df["UnitPrice"] * df["Quantity"]
    return df

@st.cache_resource(max_entries=1)
def load_KMeans():
    kmeans = KMeans()
    return kmeans

#########################
####### GLOBAL OBJECTS
#########################

data = load_clean_data()


#########################
####### UI
#########################

st.title("Customer Clustering Dashboard")

with st.expander(label="Filter Data",
                 icon=":material/filter_list:"):
    country_filter = st.multiselect(label="Filter by country:",
                                options=data["Country"].unique(), 
                                default=None)

if country_filter:
    show_data = data[data["Country"].isin(country_filter)]
else:
    show_data = data

    
agg_data = show_data.groupby("CustomerID", as_index=False)\
    .agg(
        LTDValue=("Revenue", "sum"),
        PurchaseFrequency = ("InvoiceNo", "nunique"),
        MostRecentPurchase = ("InvoiceDate", "max")
        )
    
max_date = pd.to_datetime(data["InvoiceDate"]).max()

agg_data["PurchaseRecency"] = (max_date - pd.to_datetime(agg_data["MostRecentPurchase"])).dt.days
agg_data["CustomerID"] = agg_data["CustomerID"].astype(int).astype(str)

top_item = show_data["Description"].mode()
if len(top_item) != 1:
    top_item = "Multiple most popular items"
else:
    top_item = top_item.item()

avg_rev = round(show_data["Revenue"].mean(),2)


c1, c2 = st.columns(2)

with c1.container(border=True):
    st.metric(label="Most Popular Item",
            value=top_item.title(),
            delta=+1)
    
with c2.container(border=True):
    st.metric(label="Average Revenue",
              value=f"${avg_rev}",
              delta=+1)


c1, c2 = st.columns([0.55,0.45])
with c1:
    st.subheader("Cleaned Data")
    st.dataframe(show_data[["InvoiceNo", "CustomerID", "StockCode",\
    "Description", "Quantity", "UnitPrice", "Revenue"]],
                 hide_index=True,
                 use_container_width=True)
with c2:
    st.subheader("Grouped Data")
    st.dataframe(agg_data[["CustomerID", "LTDValue", "PurchaseFrequency", "PurchaseRecency"]],
                 hide_index=True,
                 use_container_width=True)
