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
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["Year"] = df["InvoiceDate"].dt.year.astype(str)
    return df

#########################
####### LOAD DATA
#########################
data = load_clean_data()

countries = data["Country"].unique()

max_date = pd.to_datetime(data["InvoiceDate"]).max()

years = data["InvoiceDate"].dt.year.unique().astype(str)

years = np.append("All", years)


#########################
####### UI
#########################
st.title("Compare Customers & Orders Across Countries")

with st.expander(label="Filters"):
    country_filter = st.multiselect(label= "Filter by country",
                                    options=countries, 
                                    default= None)

    year_filter = st.selectbox(label="Filter by year",
                            options=years)

#Conditional filtering
if country_filter:
    country_data = data[data["Country"].isin(country_filter)]
else:
    country_data = data
    
if year_filter == "All":
    show_data = country_data
else:
    show_data = country_data[country_data["Year"] == year_filter]

#Create country list
if country_filter:
    display_countries = ", ".join(country_filter)
else:
    display_countries = "All"
   
   
#Aggregate data
agg_data = show_data.groupby("CustomerID", as_index=False)\
    .agg(
        LTDValue=("Revenue", "sum"),
        PurchaseFrequency = ("InvoiceNo", "nunique"),
        MostRecentPurchase = ("InvoiceDate", "max")
        )

#Calculated/Adjusted Fields
agg_data["PurchaseRecency"] = (max_date - pd.to_datetime(agg_data["MostRecentPurchase"])).dt.days
agg_data["CustomerID"] = agg_data["CustomerID"].astype(int).astype(str)

#Calculate top item
top_item = show_data["Description"].mode()
if len(top_item) != 1:
    top_item = "Multiple most popular items"
else:
    top_item = top_item.item()

#Calculate mean revenue
avg_rev = round(show_data["Revenue"].mean(),2)


#Conditional metric cards
c1, c2 = st.columns(2)

if year_filter == "2011":
    
    prev_top_item = country_data[country_data["Year"]=="2010"]["Description"].mode()
    
    if len(prev_top_item):
        prev_top_item = "Previous Year: Multiple"
    else:
        prev_top_item = prev_top_item.item()
    
    with c1.container(border=True):
        st.metric(label="Most Popular Item",
                value=top_item.title(),
                delta=prev_top_item,
                delta_color="off",
                )
    
    prev_rev = round(country_data[country_data["Year"]=="2010"]["Revenue"].mean(),2)
    pct_change = str(round(((avg_rev-prev_rev)/prev_rev) * 100, 2)) + "%"
    with c2.container(border=True):
        st.metric(label="Average Revenue",
                value=f"${avg_rev}",
                delta= pct_change)
else:
    with c1.container(border=True):
        st.metric(label="Most Popular Item",
                value=top_item.title())
    
    with c2.container(border=True):
        st.metric(label="Average Revenue",
            value=f"${avg_rev}")


# lineplot = px.line(show_data, x="InvoiceDate", y="Revenue")
# st.plotly_chart(lineplot)

#Display data tables
with st.expander(label="Cleaned Data"):
    st.subheader(f"Cleaned Data | Country: {display_countries} | Year: {year_filter}")
    st.dataframe(show_data[["InvoiceNo", "CustomerID", "StockCode",\
    "Description", "Quantity", "UnitPrice", "Revenue"]],
                 hide_index=True,
                 use_container_width=True)
with st.expander(label="Grouped Data (By Customer)"):
    st.subheader(f"Grouped Data (By Customer)| Country: {display_countries} | Year: {year_filter}")
    st.dataframe(agg_data[["CustomerID", "LTDValue", "PurchaseFrequency", "PurchaseRecency"]],
                 hide_index=True,
                 use_container_width=True)
