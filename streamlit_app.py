import streamlit as st
import pandas as pd
import numpy as np
import datetime as dt
import plotly.express as px
from sklearn.cluster import KMeans

#########################
####### SETUP
#########################
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
    df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
    return df

#########################
####### LOAD DATA
#########################

data = load_clean_data()

country_choice = data["Country"].unique()

max_date = pd.to_datetime(data["InvoiceDate"]).max()

#########################
####### UI
#########################

st.title("🌏 Compare Customers & Orders Across Countries")

country_filter = st.multiselect(label= "Filter ⚙️⚙️⚙️",
                                options=country_choice, 
                                default= None)

show_data = data[data["Country"].isin(country_filter)]
   
agg_data = show_data.groupby("CustomerID", as_index=False)\
    .agg(
        LTDValue=("Revenue", "sum"),
        PurchaseFrequency = ("InvoiceNo", "nunique"),
        MostRecentPurchase = ("InvoiceDate", "max")
        )

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
            value=top_item.title())
    
with c2.container(border=True):
    st.metric(label="Average Revenue",
              value=f"${avg_rev}")


with st.expander(label="Cleaned Data"):
    st.subheader(f"Cleaned Data | Country: {country_filter}")
    st.dataframe(show_data[["InvoiceNo", "CustomerID", "StockCode",\
    "Description", "Quantity", "UnitPrice", "Revenue"]],
                 hide_index=True,
                 use_container_width=True)
with st.expander(label="Grouped Data (By Customer)"):
    st.subheader(f"Grouped Data (By Customer)| Country: {country_filter}")
    st.dataframe(agg_data[["CustomerID", "LTDValue", "PurchaseFrequency", "PurchaseRecency"]],
                 hide_index=True,
                 use_container_width=True)
