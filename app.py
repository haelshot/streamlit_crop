import random
import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.preprocessing import image as kri


# Load the LSTM model
lstm_model = load_model('lstm_model.keras')

cnn_model = load_model('model.keras')

CLASS_NAMES = ['corn_blight', 'corn_common_rust', 'corn_gray_leaf_spot', 'healthy', 'no_leaves', 'rice_bacterial_leaf_blight', 'rice_blast', 'rice_brownspot']
IMAGE_SIZE = 128

# Function to preprocess input data
def preprocess_input(data):
    data_reshaped = data.reshape((1, 1, 10))  # Assuming data has 10 features
    return data_reshaped


def preprocess_image(image_path):
    # Read the image from file
    image = tf.io.read_file(image_path)
    # Decode the image to a tensor
    image = tf.image.decode_image(image, channels=3)
    # Resize image to the required input shape using TensorFlow
    resized_image = tf.image.resize(image, (64, 64))
    # Convert to numpy array
    img_array = resized_image.numpy()
    # Normalize pixel values to be between 0 and 1
    img_array = img_array / 255.0
    # Expand dimensions to match model input shape (assuming your model expects a batch dimension)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


# Streamlit App
def main():
    st.title("Rice/Maize Prediction and Classification App")

    # Sidebar navigation
    page = st.sidebar.selectbox("Select a page", ["Prediction","Classification", "About"])

    if page == "Prediction":
        st.header("LSTM Model Prediction")

        # Input widgets for each feature
        year = st.number_input("Year", min_value=0)
        month = st.number_input("Month", min_value=1, max_value=12)
        max_temp = st.number_input("Max Temperature")
        min_temp = st.number_input("Min Temperature")
        rel_humidity = st.number_input("Relative humidity")
        rainfall_mm = st.number_input("Rainfall Length (mm)")
        wind_speed = st.number_input("Wind Speed")
        sunshine_hours = st.number_input("Sunshine hours", min_value=1, max_value=24)
        evaporation_mm = st.number_input("Evaporation Length (mm)")

        # Create a DataFrame with the input data
        input_data = pd.DataFrame({
            'Year': [year],
            'Month': [month],
            'max_temp': [max_temp],
            'min_temp': [min_temp],
            'rel_humidity': [rel_humidity],
            'rel_humidity.1': [rel_humidity],
            'rainfall_mm': [rainfall_mm],
            'wind_speed': [wind_speed],
            'sunshine_hours': [sunshine_hours],
            'evaporation_mm': [evaporation_mm],
        })

        # Preprocess input data
        X_input = input_data.values
        X_input_processed = preprocess_input(X_input)

        if st.button("Predict"):
            # Make predictions using LSTM model
            lstm_prediction = lstm_model.predict(X_input_processed)[0, 0]
            t = True if lstm_prediction == 1 else False
            st.subheader("LSTM Prediction:")
            st.write(f"The prediction is {t}")
            d = random.choice(['Disease', 'Pest'])
            if lstm_prediction == 1:
                st.write(f"There is a possibility of {d} attacking the crop")
            else:
                st.write(f"The crop is safe for the selected time/year and given parameters")

    elif page == "About":
        st.header("About This App")
        st.write("This app uses an LSTM model to predict clusters based on input features.")
        st.write("This app uses an CNN model to classify images based on input images.")

    elif page == "Classification":
        st.header("Classification Page") 
        uploaded_file = st.file_uploader("Choose a crop image...", type="jpg")

        if uploaded_file is not None:
            # Display the uploaded image
            img = kri.load_img(uploaded_file, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            st.image(img, caption="Uploaded Image.", use_column_width=True)
            
            img_array = kri.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)

            if st.button("Classify"):
                # Make predictions using CNN model
                prediction = cnn_model.predict(img_array)
                predicted_class_index = np.argmax(prediction)
                predicted_class = CLASS_NAMES[predicted_class_index]
                st.write(f"The selected image is in the following class: {predicted_class}")

if __name__ == "__main__":
    main()
