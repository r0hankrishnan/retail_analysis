def load_data():
    import pandas as pd
    df = pd.read_excel('../data/online_retail.xlsx')
    return df


def load_clean_data():
    import pandas as pd
    try:
        df = pd.read_csv('../data/clean_retail.csv', index_col=0)
    except Exception as e:
        print(f"Error: {e}")
        
    print("Data loaded successfully.")
    return df