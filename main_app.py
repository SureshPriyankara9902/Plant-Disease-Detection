# Library imports
import numpy as np
import streamlit as st
import cv2
from keras.models import load_model
import tensorflow as tf
import time
import plotly.graph_objects as go

# Loading the Model
model = load_model('plant_disease_model.h5')

# Name of Classes
CLASS_NAMES = ('Tomato-Bacterial_spot', 'Potato-Barly blight', 'Corn-Common_rust')

# Set page config
st.set_page_config(page_title="Plant Disease Detection", page_icon="🌿", layout="wide")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        background-color: #e9f5e9;
        color: #333;
        font-family: 'Arial', sans-serif;
        margin: 0;
        padding: 0;
    }
    .main {
        background-color: #fff;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin: 0 auto;
        max-width: 1200px;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        margin: 10px 5px;
        cursor: pointer;
        border-radius: 8px;
        transition: background-color 0.3s ease;
        height: 50px;
        width: 150px;
        justify-content: center;
    }
    .stButton>button:hover {
        background-color:blue;
        color:white;
    }
    .stButton.reset_button {
        color: white;
        background-color: blue;
        transition: background-color 0.3s ease;
        justify-content: center;
    }
    .stButton.reset_button>button:hover {
        background-color: white;
    }
    .stFileUploader>div>label {
        font-size: 20px;
        color: #4CAF50;
    }
    .stFileUploader>div>input[type="file"] {
        font-size: 18px;
        padding: 15px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        cursor: pointer;
    }
    .stProgress > div > div > div {
        border-radius: 10px;
        background-color: #eee; /* Background color of the progress bar */
    }
    .stProgress > div > div > div > div {
        border-radius: 10px;
        background-color: green; /* Red color for the filled portion of the progress bar */
    }
    .centered {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 100%;
        text-align: center;
    }
    .button-container {
        display: flex;
        justify-content: center;
        margin-top: 20px;
    }
    .header {
        text-align: center;
        padding: 2px;
        margin-top: 0px;
        font-size: 24px;
        color: #4CAF50;
        font-weight: bold;
    }
    
    </style>
    """, unsafe_allow_html=True)

# Sidebar for input widgets
with st.sidebar:
    st.title("🌿Plant Disease Detection🔎")
    st.markdown('<h1 style="text-align: center;">🌱🐛🐛</h1>', unsafe_allow_html=True)
    st.markdown("### Upload an image file of the plant leaf to detect the disease")

    # File uploader widget
    uploaded_file = st.file_uploader("Choose a file...", type=['jpg', 'jpeg', 'png'])
    
    # Buttons for prediction and reset
    submit_button = st.button('Predict Disease')
    reset_button = st.button('Reset')

# Main container
with st.container():
    st.markdown('<h1 style="text-align: center;">🌿Plant🌱 Disease🐛 Detection🔎🌿</h1>', unsafe_allow_html=True)

    if reset_button:
        st.empty()

    if submit_button:
        if uploaded_file is not None:
            try:
                file_type = uploaded_file.type

                # Handle image files
                if file_type in ['image/jpeg', 'image/png']:
                    # Convert the file to an opencv image.
                    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
                    opencv_image = cv2.imdecode(file_bytes, 1)
                    
                    # Display the image
                    st.markdown('<div class="centered">', unsafe_allow_html=True)
                    st.image(opencv_image, channels="BGR", caption='Uploaded Image', use_column_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Resizing the image
                    opencv_image_resized = cv2.resize(opencv_image, (256, 256))
                    
                    # Convert image to 4 Dimension
                    opencv_image_resized = np.expand_dims(opencv_image_resized, axis=0)
                    
                    # Show progress bar and status updates
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i in range(100):
                        time.sleep(0.05)
                        progress_bar.progress(i + 1)
                        status_text.text(f"Processing: {i + 1}%")
                    
                    progress_bar.progress(100)
                    status_text.text("Processing Complete!")
                    
                    # Make Prediction
                    tf.keras.backend.clear_session()
                    try:
                        Y_pred = model.predict(opencv_image_resized)[0]
                        # Display prediction result
                        result = CLASS_NAMES[np.argmax(Y_pred)]
                        st.markdown(f"<h2 style='text-align: center;'>This is a {result.split('-')[0]} leaf with {result.split('-')[1]}</h2>", unsafe_allow_html=True)
                        
                        # Plot prediction probabilities
                        fig = go.Figure()
                        fig.add_trace(go.Bar(x=CLASS_NAMES, y=Y_pred, marker=dict(color=['#FF9999', '#FF6666', '#FF3333'])))
                        fig.update_layout(title='Prediction Probabilities',
                                          xaxis_title='Classes',
                                          yaxis_title='Probability',
                                          plot_bgcolor='rgba(0,0,0,0)',
                                          paper_bgcolor='rgba(0,0,0,0)')
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error in model prediction: {e}")
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
        else:
            st.warning("Please upload a file before clicking 'Predict Disease'.")
