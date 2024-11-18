import streamlit as st
import pandas as pd
import joblib

# Load the machine learning pipeline (already saved)
gbr_pipeline = joblib.load('gbr_pipeline.pkl')

# Function to calculate the features as mentioned in the problem
def calculate_features(df, workingday_counts, non_workingday_counts):
    # Set 'yr' automatically to 2 (for 2013)
    df['yr'] = 2  # Set year for 2013 (encoded as 2)

    # 'dry_precip' feature
    # Update to account for 4 weather situations
    df['dry_precip'] = df['weathersit'].apply(lambda x: 1 if x in [1, 2] else 2)

    # Hourly avg workingday and non-workingday calculation
    def map_hourly_avg(row):
        if row['workingday'] == 1:
            return workingday_counts.get((row['mnth'], row['hr']), 0)
        else:
            return non_workingday_counts.get((row['mnth'], row['hr']), 0)

    df['hourly_avg_workingday'] = df.apply(lambda row: map_hourly_avg(row) if row['workingday'] == 1 else 0, axis=1)
    df['hourly_avg_nonworkingday'] = df.apply(lambda row: map_hourly_avg(row) if row['workingday'] == 0 else 0, axis=1)

    # Prepare the final features for prediction, including 'temp_expected_1' for the model
    return df[['yr', 'mnth', 'hum', 'hourly_avg_workingday', 'hourly_avg_nonworkingday', 'temp_expected_1', 'dry_precip']]

# Streamlit interface
st.title('Bike Usage Prediction for 2013')

# Upload CSV files for hourly averages
workingday_csv = st.sidebar.file_uploader("Upload Working Day Averages CSV", type=["csv"])
non_workingday_csv = st.sidebar.file_uploader("Upload Non-Working Day Averages CSV", type=["csv"])

# Check if files are uploaded
if workingday_csv is not None and non_workingday_csv is not None:
    # Load the CSV files into DataFrames
    workingday_counts = pd.read_csv(workingday_csv)
    non_workingday_counts = pd.read_csv(non_workingday_csv)

    # Convert them to dictionaries for easier access (if needed)
    workingday_counts_dict = workingday_counts.set_index(['mnth', 'hr'])['cnt'].to_dict()
    non_workingday_counts_dict = non_workingday_counts.set_index(['mnth', 'hr'])['cnt'].to_dict()

    # Input form for users
    st.sidebar.header('Input Parameters')
    temp_expected_1 = st.sidebar.number_input('Temperature expected for the next hour', min_value=-20, max_value=50, value=20)
    mnth = st.sidebar.selectbox('Month', options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], index=0)
    hr = st.sidebar.selectbox('Hour', options=[i for i in range(24)], index=0)
    workingday = st.sidebar.selectbox('Working Day', options=[0, 1], index=1)  # 1 for working day, 0 for non-working day
    hum = st.sidebar.number_input('Humidity', min_value=0, max_value=100, value=50)
    weathersit = st.sidebar.selectbox('Weather Situation', options=[1, 2, 3, 4], index=0)  # 1: Clear, 2: Cloudy, 3: Rain, 4: Snow (or other)

    # Create a small dummy dataframe for input (without 'cnt' column)
    input_data = pd.DataFrame({
        'temp_expected_1': [temp_expected_1],  # Ensure correct column name
        'mnth': [mnth],
        'hr': [hr],
        'workingday': [workingday],
        'hum': [hum],
        'weathersit': [weathersit],
    })

    # Automatically set 'yr' to 2 for 2013 (encoded as 2)
    input_data['yr'] = 2

    # Calculate features for input data
    input_data = calculate_features(input_data, workingday_counts_dict, non_workingday_counts_dict)

    # Standardize column names (ensure proper capitalization - no capitals)
    input_data.columns = input_data.columns.str.lower()

    # Display the input features
    st.subheader('Input Features')
    st.write(input_data)

    # Predict using the loaded model
    prediction = gbr_pipeline.predict(input_data)

    # Display the prediction
    st.subheader('Predicted Bike Usage')
    st.write(f"Predicted bike usage for the given hour and month: {prediction[0]:.2f} bikes")

else:
    st.write("Please upload the CSV files containing the hourly averages for working days and non-working days.")