import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the trained Logistic Regression model from the pickle file
model = pickle.load(open('model.pkl', 'rb'))

# Title of the web app
st.title("Placement Prediction Based on IQ and CGPA")

# Function to plot decision region
def plot_decision_region(X, y, model):
    # Create a mesh grid based on the range of input features
    x_min, x_max = X['CGPA'].min() - 1, X['CGPA'].max() + 1
    y_min, y_max = X['IQ'].min() - 10, X['IQ'].max() + 10
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.5))
    
    # Flatten the mesh grid and create input data for predictions
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Plot the decision boundary
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.RdYlBu)
    plt.scatter(X['CGPA'], X['IQ'], c=y, edgecolors='k', marker='o', s=100, cmap=plt.cm.RdYlBu)
    plt.xlabel('CGPA')
    plt.ylabel('IQ')
    plt.title('Decision Region')

# Function to get user inputs
def user_input_features():
    # Input fields in the sidebar for user to enter CGPA and IQ
    CGPA = st.sidebar.number_input("Enter CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
    IQ = st.sidebar.number_input("Enter IQ", min_value=70, max_value=160, value=100, step=1)
    
    # Creating a DataFrame for input data
    data = {'CGPA': [CGPA], 'IQ': [IQ]}
    return pd.DataFrame(data)

# Get user inputs
input_df = user_input_features()

# Dummy data for the plot (replace with your actual dataset for better visualization)
# For example purposes, using random CGPA and IQ values
X_train = pd.DataFrame({
    'CGPA': np.random.uniform(5.0, 10.0, size=100),
    'IQ': np.random.uniform(80, 150, size=100)
})
y_train = np.random.randint(0, 2, size=100)

# Plot the decision region at the top
st.write("### Decision Region:")
fig, ax = plt.subplots()
plot_decision_region(X_train, y_train, model)
st.pyplot(fig)

# Display the user input data
st.write("### Input Data:")
st.write(input_df)

# Predict based on the model and user inputs
if st.button('Predict Placement'):
    # Make the prediction using the trained model
    prediction = model.predict(input_df)
    
    # Display the result based on the prediction
    if prediction[0] == 1:
        st.success("Placement will happen! 🎉")
    else:
        st.warning("Placement will not happen. 😔")
