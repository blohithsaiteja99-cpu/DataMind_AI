from io import BytesIO
import pandas as pd
MAX_UPLOAD_BYTES=25*1024*1024
MAX_ROWS=250_000

def read_dataset(filename: str, content: bytes) -> pd.DataFrame:
    if len(content)>MAX_UPLOAD_BYTES: raise ValueError("File is too large. Maximum upload size is 25 MB.")
    name=filename.lower()
    if name.endswith(".csv"): df=pd.read_csv(BytesIO(content))
    elif name.endswith((".xlsx",".xls")): df=pd.read_excel(BytesIO(content))
    else: raise ValueError("Unsupported file type. Upload CSV or Excel.")
    if df.empty: raise ValueError("The uploaded file has no data rows.")
    if len(df)>MAX_ROWS: raise ValueError(f"Dataset exceeds the {MAX_ROWS:,}-row limit.")
    if len(df.columns)>200: raise ValueError("Dataset exceeds the 200-column limit.")
    df.columns=[str(c).strip() or f"column_{i}" for i,c in enumerate(df.columns)]
    if df.columns.duplicated().any(): raise ValueError("Column names must be unique. Rename duplicate columns.")
    return df
