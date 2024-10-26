import pandas as pd

def load_data():
    import pandas as pd
    path = "./data/online_retail.xlsx"
    data = pd.read_excel(path)
    return data

