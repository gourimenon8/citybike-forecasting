import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Disable Hopsworks "model serving" client
os.environ["HOPSWORKS_DISABLE_SERVING"] = "true"
import hopsworks

# Set page configuration
st.set_page_config(
    page_title="Citi Bike Ride Forecast",
    page_icon="🚲",
    layout="wide"
)

# Add this near the top of your streamlit_inference.py file
# Set MLflow credentials
mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])

# If you need authentication with DAGsHub/MLflow
os.environ["MLFLOW_TRACKING_USERNAME"] = st.secrets["MLFLOW_TRACKING_USERNAME"]
os.environ["MLFLOW_TRACKING_PASSWORD"] = st.secrets["MLFLOW_TRACKING_PASSWORD"]
os.environ["MLFLOW_TRACKING_URI"] = st.secrets["MLFLOW_TRACKING_URI"]

print(os.environ["MLFLOW_TRACKING_USERNAME"])
print(os.environ["MLFLOW_TRACKING_PASSWORD"])
print(os.environ["MLFLOW_TRACKING_URI"])

# Page header
st.title("🚲 Citi Bike Station Ride Forecasting")
st.markdown("### Predict tomorrow's ride counts for New York City stations")

# Cache the Hopsworks connection
@st.cache_resource
def get_hopsworks_connection():
    st.info("🔌 Connecting to Hopsworks...")
    project = hopsworks.login(api_key_value=st.secrets["HOPSWORKS_API_KEY"])
    fs = project.get_feature_store()
    return project, fs

# Cache the feature data loading - add underscore to prevent hashing
@st.cache_data(ttl=3600)
def load_feature_data(_fs):  # Added underscore here
    fg = _fs.get_feature_group("citibike_daily_rides", version=1)
    df = fg.read()
    return df

# Function to clean station names for model names
def clean_model_name(station):
    return station.replace(" ", "_").replace("&", "").replace("/", "_")

# Function to create lag features
def make_lags(df, stations, n_lags=28):
    next_day = df.index.max() + timedelta(days=1)
    tmp = df.copy()
    tmp.loc[next_day] = np.nan
    
    # Create lag features only for selected stations
    X = pd.concat({
        f"{station}_lag{lag}": tmp[station].shift(lag).astype("float32")
        for station in stations
        for lag in range(1, n_lags + 1)
    }, axis=1)
    
    return X.iloc[[-1]].astype("float32")

# Connect to Hopsworks and load data
try:
    project, fs = get_hopsworks_connection()
    df = load_feature_data(fs)
    
    # Pivot the data
    daily = (
        df
        .pivot(index="date", columns="start_station_name", values="ride_count")
        .fillna(0)
    )
    daily.index = pd.to_datetime(daily.index)
    
    # Get list of all stations
    all_stations = sorted(daily.columns.tolist())
    
    # Create sidebar for settings
    st.sidebar.header("Settings")
    
    # Station selection
    selected_stations = st.sidebar.multiselect(
        "Select stations to forecast:",
        options=all_stations,
        default=all_stations[:5] if len(all_stations) >= 5 else all_stations
    )
    
    if not selected_stations:
        st.warning("Please select at least one station")
        st.stop()
    
    # Display historical data tab and forecast tab
    tab1, tab2 = st.tabs(["Historical Data", "Forecasts"])
    
    with tab1:
        st.subheader("Historical Ride Data")
        
        # Date range slider
        date_range = st.slider(
            "Select date range",
            min_value=daily.index.min().to_pydatetime(),
            max_value=daily.index.max().to_pydatetime(),
            value=(daily.index.max() - timedelta(days=30), daily.index.max().to_pydatetime())
        )
        
        # Filter data based on date range
        mask = (daily.index >= date_range[0]) & (daily.index <= date_range[1])
        filtered_data = daily.loc[mask, selected_stations]
        
        # Plot historical data
        fig, ax = plt.subplots(figsize=(12, 6))
        for station in selected_stations:
            ax.plot(filtered_data.index, filtered_data[station], label=station)
        
        ax.set_xlabel("Date")
        ax.set_ylabel("Number of Rides")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        
        # Show data table
        st.dataframe(filtered_data)
    
    with tab2:
        st.subheader("Ride Forecasts")
        
        # Get last 28 days for feature creation
        last28 = daily.tail(28)
        
        with st.spinner("Generating forecasts..."):
            # Create prediction dataframe
            predictions = []
            
            # Connect to MLflow
            mlflow.set_tracking_uri(st.secrets["MLFLOW_TRACKING_URI"])
            client = mlflow.tracking.MlflowClient()
            
            # Create lag features for prediction
            X_tomorrow = make_lags(daily, selected_stations)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("### Tomorrow's Predictions")
                
                # Load models and make predictions
                for station in selected_stations:
                    clean_name = clean_model_name(station)
                    model_name = f"citibike_model_{clean_name}"
                    
                    with st.status(f"Loading model for {station}...") as status:
                        try:
                            # Get latest registered version
                            versions = client.get_latest_versions(model_name)
                            if not versions:
                                st.error(f"No model found for {station}")
                                continue
                                
                            uri = versions[0].source
                            model = mlflow.pyfunc.load_model(uri)
                            
                            # Filter features for this station
                            station_features = [col for col in X_tomorrow.columns if col.startswith(f"{station}_lag")]
                            
                            # If the model is using top10 features, we need to load those
                            try:
                                # Try to get model artifacts location
                                model_path = versions[0].source.replace("models:/", "").split("/")[0]
                                top10_path = os.path.join(mlflow.get_tracking_uri(), "artifacts", model_path, "top10_features.json")
                                
                                with open(top10_path) as f:
                                    top10_features = json.load(f)
                                    X_selected = X_tomorrow[top10_features]
                            except:
                                # If we can't find top10 features, use all features
                                X_selected = X_tomorrow
                            
                            pred = model.predict(X_selected)[0]
                            pred_rounded = round(pred)
                            
                            # Add to predictions list
                            tomorrow = daily.index.max() + timedelta(days=1)
                            predictions.append({
                                "station": station,
                                "date": tomorrow,
                                "prediction": pred_rounded,
                                "actual": None  # We don't have actual values for tomorrow yet
                            })
                            
                            # Show success
                            status.update(label=f"✅ {station}: {pred_rounded} rides tomorrow", state="complete")
                            
                        except Exception as e:
                            status.update(label=f"⚠️ Error predicting for {station}: {str(e)}", state="error")
            
            with col2:
                # Create a prediction bar chart
                if predictions:
                    st.write("### Prediction Visualization")
                    pred_df = pd.DataFrame(predictions)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.barplot(x="prediction", y="station", data=pred_df, ax=ax)
                    ax.set_title("Predicted Rides for Tomorrow")
                    ax.set_xlabel("Number of Rides")
                    ax.set_ylabel("Station")
                    st.pyplot(fig)
                    
                    # Show predictions table
                    st.write("### Prediction Details")
                    display_df = pred_df[["station", "prediction"]].set_index("station")
                    st.dataframe(display_df)
                    
                    # Download predictions as CSV
                    csv = pred_df.to_csv(index=False)
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=csv,
                        file_name=f"citibike_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No predictions could be made.")
    
except Exception as e:
    st.error(f"An error occurred: {str(e)}")
    st.error("Please check your Hopsworks and MLflow credentials in the .streamlit/secrets.toml file.")