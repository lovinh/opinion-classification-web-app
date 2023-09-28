import pandas as pd

def read_file(file_path : str, file_type : str) -> pd.DataFrame:
    data = None
    if (file_type == "xlsx"):
        data = pd.read_excel(file_path)
    elif (file_type == "csv"):
        data = pd.read_csv(file_path)
    else: 
        raise TypeError("Invalid File Type")
    return data

if __name__ == "__main__":
    print(read_file(f"app/static/files/test.xlsx", "xlsx").head())
