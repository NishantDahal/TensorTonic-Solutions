import pandas as pd

def inspect_dataframe(data: dict) -> dict:
    """
    Returns rows, cols, columns, dtypes, and total_values in a dictionary.
    """
    df = pd.DataFrame(data)
    d = {}
    d["rows"] = df.shape[0]
    d["cols"] = df.shape[1]

    d["columns"] = df.columns.tolist()
    d["dtypes"] = {k:str(v.dtype) for k, v in df.items()}
    d["total_values"] = df.size

    return d