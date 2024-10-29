def load_data():
    import pandas as pd
    df = pd.read_excel('../data/raw/online_retail.xlsx')
    return df


def load_clean_data():
    import pandas as pd
    try:
        df = pd.read_csv('../data/cleaned/clean_retail.csv', index_col=0)
    except Exception as e:
        print(f"Error: {e}")
        
    print("Data loaded successfully.")
    return df

def load_analysis_data(relPath:str):
    import pandas as pd
    
    try:
        df = pd.read_csv(f"../data/analysis{relPath}", index_col=0)
    except Exception as e:
        print(f"Error: {e}")
        
    if "Unnamed: 0" in df.columns:
        df = df.drop("Unnamed: 0", axis = 1)
    
    return df