import pandas as pd

def create_dataframe(data: dict) -> dict:
    """
    Returns a dictionary with data, shape [rows, columns], and ordered column names.
    """
    df = pd.DataFrame(data)
    d = {}
    d["data"] = df.to_dict('list')
    d["shape"] = list(df.shape)
    d["columns"] = list(df.keys())
    return d