import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Your Streamlit app content
st.title("Flood Prediction Project Using ML")
st.write("Welcome to the Flood Prediction Web App!")
# Add the line at the end of the app
st.write("---")  # This adds a horizontal line as a separator
st.markdown("### ML Project by Sayma M, MCA II")

# Upload the datasets
st.write("Upload your rainfall dataset:")
uploaded_rain = st.file_uploader("Choose Rainfall CSV file", type="csv")
st.write("Upload your river level dataset:")
uploaded_river = st.file_uploader("Choose River Level CSV file", type="csv")

if uploaded_rain is not None and uploaded_river is not None:
    # Load the CSV files into DataFrames
    df_rain = pd.read_csv(uploaded_rain)
    df_river = pd.read_csv(uploaded_river)

    # Display the datasets
    st.write("Rainfall Dataset:")
    st.write(df_rain.head())
    
    st.write("River Level Dataset:")
    st.write(df_river.head())

    # Plot the rainfall
    st.write("Rainfall Plot:")
    df_rain.plot(x='Date/Time', y='Cumulative rainfall (mm)', style='o')
    plt.title('Rainfall')
    plt.xlabel('Date')
    plt.ylabel('Rainfall in mm')
    st.pyplot(plt)

    # Plot the river level
    st.write("River Level Plot:")
    df_river.plot(x='Date/Time', y='Level (m)', style='o')
    plt.title('River Level')
    plt.xlabel('Date')
    plt.ylabel('Max Level')
    st.pyplot(plt)

    # Merge the datasets on Date/Time
    df = pd.merge(df_rain, df_river, how='outer', on=['Date/Time'])

    # Fill missing values
    df['Cumulative rainfall (mm)'] = df['Cumulative rainfall (mm)'].fillna(0)
    df['Level (m)'] = df['Level (m)'].fillna(0)

    # Drop unnecessary columns
    df = df.drop(columns=['Current rainfall (mm)', 'Date/Time'])

    # Split features and target
    X = df.iloc[:, :1].values  # Rainfall
    y = df.iloc[:, 1:2].values  # River Level

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Train the linear regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Plot training set results
    plt.scatter(X_train, y_train)
    plt.plot(X_train, regressor.predict(X_train), color='red')
    plt.title('Rainfall vs River Level (Training set)')
    plt.xlabel('Rainfall')
    plt.ylabel('River Level')
    st.pyplot(plt)

    # Plot test set results
    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_train, regressor.predict(X_train), color='blue')
    plt.title('Rainfall vs River Level (Test set)')
    plt.xlabel('Rainfall')
    plt.ylabel('River Level')
    st.pyplot(plt)

    # Predict river level based on user input
    st.write("Enter Rainfall amount in mm to predict river level:")
    Rainfall_Amount = st.number_input("Rainfall Amount (mm)", min_value=0)

    if Rainfall_Amount:
        predicted_riverlevel = regressor.predict([[Rainfall_Amount]])
        st.write(f"Predicted River Level: {predicted_riverlevel[0][0]} meters")

        # Flood prediction logic
        if predicted_riverlevel > 1.5:
            st.write("FLOOD WARNING!")
        else:
            st.write("No flood expected.")
