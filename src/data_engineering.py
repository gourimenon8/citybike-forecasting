"""
Download, clean, and upload Citi Bike data to Hopsworks
"""

"""
Download, clean, and upload Citi Bike data to Hopsworks
"""


import os
import pandas as pd
import requests
from zipfile import ZipFile, BadZipFile
from datetime import datetime
import hopsworks



def load_months_data(months, base_url="https://s3.amazonaws.com/tripdata"):
    all_dfs = []
    for month in months:
        url = f"{base_url}/{month}-citibike-tripdata.csv.zip"
        zip_path = f"data/{month}.zip"
        print(f"📥 Downloading {month} data from {url}...")
        r = requests.get(url)
        if r.status_code != 200:
            raise RuntimeError(f"Failed to download {url} (status {r.status_code})")
        with open(zip_path, "wb") as f:
            f.write(r.content)
        try:
            with ZipFile(zip_path, "r") as z:
                z.extractall(".")
        except BadZipFile:
            raise RuntimeError(f"Downloaded file {zip_path} is not a valid zip archive")
        csv_file = next(f for f in os.listdir() if f.endswith(".csv") and month in f)
        print(f"    Extracted {csv_file}")
        df_month = pd.read_csv(csv_file, low_memory=False)
        all_dfs.append(df_month)
    df = pd.concat(all_dfs, ignore_index=True)
    print(f"✅ Combined raw data shape: {df.shape}")
    return df


def aggregate_top_stations(df, top_n=3):
    # Lowercase columns & parse datetime
    df.columns = df.columns.str.lower()
    df["started_at"] = pd.to_datetime(df["started_at"])
    df = df.dropna(subset=["start_station_name"])
    # Top N stations
    top_stations = (
        df["start_station_name"]
        .value_counts()
        .nlargest(top_n)
        .index
        .tolist()
    )
    print(f"🏙️ Top {top_n} stations: {top_stations}")
    df = df[df["start_station_name"].isin(top_stations)].copy()
    # Use Python date objects (no conversion to Timestamp)
    df["date"] = df["started_at"].dt.date
    agg_df = (
        df.groupby(["start_station_name", "date"])
          .size()
          .reset_index(name="ride_count")
    )
    # Do NOT convert date to datetime64 — keep as Python date
    # Cast ride_count to int64 for bigint compatibility
    agg_df["ride_count"] = agg_df["ride_count"].astype("int64")
    # Ensure station names are strings
    agg_df["start_station_name"] = agg_df["start_station_name"].astype(str)
    print(f"✅ Aggregated data shape: {agg_df.shape}")
    return agg_df


if __name__ == "__main__":
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("Please set the HOPSWORKS_API_KEY environment variable")

    # 1️⃣ Load raw data
    months = [ "202402" ]
    raw_df = load_months_data(months)

    # 2️⃣ Aggregate
    agg_df = aggregate_top_stations(raw_df, top_n=3)

    # 3️⃣ Upload to Hopsworks
    project = hopsworks.login(api_key_value=api_key)
    fs = project.get_feature_store()
    fg = fs.get_feature_group("citibike_daily_rides", version=1)
    fg.insert(agg_df, write_options={"wait_for_job": True})
    print("✅ Data inserted into Hopsworks Feature Store")

