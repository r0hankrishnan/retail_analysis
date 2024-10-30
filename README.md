# Retail Store Analysis

Using UK online retail data from 2009 to 2011, I will conduct a hypothetical data science project starting from data exploration and ending in model development and deployment.

## Table of Contents 
1. [Data](#data)
2. [Exploration](#exploration)
3. [Cleaning](#cleaning)
4. [Feature Engineering](#feature-engineering)
5. [KMeans Clustering](#kmeans-clustering)
6. [Takeaways](takeaways)

## Data
I am using the [UK online retail data set from the UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/352/online+retail). This description page for this data lists the following information about the data: 

This is a transactional data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail. The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.

**Variables:**

| Variable Name | Role | Type | Description	Units | Missing Values |
|---------------|------|------|-------------------|----------------|
| InvoiceNo | ID | Categorical | A 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation | No |
| StockCode | ID | Categorical | A 5-digit integral number uniquely assigned to each distinct product | No |
| Description | ID | Categorical | Product name | No |
| Quantity | Feature | Integer | The quantities of each product (item) per transaction | No |
| InvoiceDate | Feature | Date | The dat and time when each transaction was generated | No |
| UnitPrice | Feature | Continuous | Product price per unit sterling | No |
| CustomerID | Feature | Categorical | A 5-digit integral number uniquely assigned ot each customer | No |
| Country | Feature | Categorical | The name of the country where each customer resides | No |

**Additional Variable Information:**
- InvoiceNo: Invoice number. Nominal, A 6-digit integral number uniquely assigned to each transaction. **If this code starts with letter 'c', it indicates a cancellation.** 
- StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
- Description: Product (item) name. Nominal.
- Quantity: The quantities of each product (item) per transaction. Numeric.	
- InvoiceDate: Invoice Date and time. Numeric, The day and time when each transaction was generated.
- UnitPrice: Unit price. Numeric, Product price per unit in sterling.
- CustomerID: Customer number. Nominal, A 5-digit integral number uniquely assigned to each customer.
- Country: Country name. Nominal, The name of the country where each customer resides.


## Exploration

## Cleaning

## Feature Engineering

## KMeans Clustering

## Takeaways
