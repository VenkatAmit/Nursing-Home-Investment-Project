import os
import pandas as pd
from pathlib import Path


def load_cost_reports(data_dir: str) -> pd.DataFrame:
    csv_files = [
        "2015CostReport.csv",
        "2016CostReport.csv",
        "2017CostReport.csv",
        "2018CostReport.csv",
        "2019CostReport.csv",
        "2020CostReport.csv",
        "2021CostReport.csv",
    ]
    dfs = {}
    for file in csv_files:
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath, low_memory=False)
            df["year"] = file[:4]
            dfs[file] = df
        else:
            print(f"File not found: {filepath}")
    if not dfs:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")
    merged = pd.concat(dfs.values(), ignore_index=True)
    print(f"Loaded {len(dfs)} files -> {merged.shape[0]:,} rows x {merged.shape[1]} columns")
    return merged


def standardise_columns(df: pd.DataFrame) -> pd.DataFrame:
    renames = {
        "Provider CCN": "ProviderCCN",
        "Facility Name": "FacilityName",
        "Street Address": "StreetAddress",
        "State Code": "StateCode",
        "Zip Code": "ZipCode",
        "Net Patient Revenue": "NetPatientRevenue",
        "Less Total Operating Expense": "LessTotalOperatingExpense",
        "Total Other Income": "TotalOtherIncome",
        "Total Income": "TotalIncome",
        "Net Income": "NetIncome",
        "Total Current Assets": "Totalcurrentassets",
        "Total current liabilities": "Totalcurrentliabilities",
        "Total fund balances": "Totalfundbalances",
        "Total liabilities": "Totalliabilities",
        "Total Assets": "TotalAssets",
        "Accounts Receivable": "AccountsReceivable",
        "Total Bed Days Available": "TotalBedDaysAvailable",
        "Total Days Total": "TotalDaysTotal",
    }
    return df.rename(columns=renames)


def impute_income_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["NetIncome", "TotalIncome", "TotalOtherIncome"]:
        if col in df.columns:
            df[col].fillna(df[col].mean(), inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ROI"] = df["NetIncome"] / df["TotalAssets"]
    df["OperatingMargin"] = (
        (df["NetPatientRevenue"] - df["LessTotalOperatingExpense"])
        / df["NetPatientRevenue"]
    )
    df["Debttoequityratio"] = df["Totalliabilities"] / df["Totalfundbalances"]
    df["CurrentRatio"] = df["Totalcurrentassets"] / df["Totalcurrentliabilities"]
    df["DaysinAccountsReceivable"] = (
        df["AccountsReceivable"] / (df["NetPatientRevenue"] / 365)
    )
    df["BedOccupancyRate"] = df["TotalDaysTotal"] / df["TotalBedDaysAvailable"]
    return df


def get_numerical_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=["int64", "float64"]).columns.tolist()


def get_categorical_cols(df: pd.DataFrame) -> list:
    return df.select_dtypes(include=["object", "category"]).columns.tolist()


def build_feature_set(data_dir: str) -> pd.DataFrame:
    df = load_cost_reports(data_dir)
    df = standardise_columns(df)
    df = impute_income_columns(df)
    df = engineer_features(df)
    return df
