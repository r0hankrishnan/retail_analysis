def load_data():
    import pandas as pd
    df = pd.read_excel('../data/online_retail.xlsx')
    return df