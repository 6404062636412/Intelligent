import streamlit as st
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
 


# Load Model 1 (Machine Learning - Numeric Data)
with open("finalized_model.pkl", "rb") as file:
    model1 = pickle.load(file)

# Load Model 2 (Neural Network - Text Data)
model2 = tf.keras.models.load_model("Disaster_modelv2.h5")

# Load Tokenizer for Model 2
with open("tokenizer.pkl", "rb") as file:
    tokenizer2 = pickle.load(file)

# Sidebar for Navigation
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("📌 Select Page", 
                        ["🏠 Home",
                         "📖 การพัฒนาโมเดล Machine Learning และ Neural Network", 
                         "📊 Model 1 Machine Learning - Purchase Prediction", 
                         "📝 Model 2 Neural Network - Disaster Tweet Classification", 
                         "📚 About Model 1", 
                         "📚 About Model 2"])

# Home Page
if page == "🏠 Home":
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>Welcome to ML & NN Prediction App</h1>", unsafe_allow_html=True)
    st.markdown("### 🤖 This app provides two types of predictions:")
    st.write("- **📊 Model 1:** Predict using a machine learning model based on numerical input.")
    st.write("- **📝 Model 2:** Predict using a neural network model based on text input.")
    st.markdown("<h3 style='text-align: center;'>Use the sidebar to navigate.</h3>", unsafe_allow_html=True)

# Development Process
elif page == "📖 การพัฒนาโมเดล Machine Learning และ Neural Network":
    st.title("แนวทางการพัฒนาโมเดล Machine Learning และ Neural Network")
    st.write("""
    ## 1. Data Preparation
             The success of machine learning and neural network models depends heavily on data preparation. 
             Proper preprocessing ensures that the data is clean, structured, and suitable for training.

    - 1.1 Numerical Data for Machine Learning
        - Handling Missing Values: Identify and manage missing values appropriately to avoid bias.
        - Normalization/Standardization: Scale numerical features to maintain consistency and improve convergence during training.
        - Feature Engineering & Selection: Extract meaningful features and remove irrelevant or redundant ones to enhance model performance.
    
    - 1.2 Text Data for Neural Networks
        - Tokenization: Convert text into sequences of tokens for numerical processing.
        - Padding & Truncation: Ensure all sequences have the same length by padding shorter ones and truncating longer ones.
        - Word Embedding: Represent words as dense numerical vectors to capture semantic meanings and improve the model’s ability to understand language patterns.
             

    ## 2.  Algorithm Theory
    - **Model 1:** Traditional Machine Learning Algorithms
         - For numerical data, the model employs Decision Trees or Random Forests:
             - Decision Tree: Uses a tree-like structure to split data based on feature values, making it easy to interpret.
             - Random Forest: An ensemble learning method that combines multiple decision trees to improve accuracy and reduce overfitting.

    - **Model 2:** Neural Networks for Text Data
         - For text classification, the model uses LSTM (Long Short-Term Memory) or GRU (Gated Recurrent Unit), which are specialized recurrent neural networks (RNNs) 
         designed for sequential data:
             - LSTM: Capable of remembering long-term dependencies, making it ideal for capturing context in text.
             - GRU: A simplified alternative to LSTM that requires fewer parameters while maintaining similar performance.
    
    ## 3. Model Development Steps
             

    - **Model 1:**
      1. Feature Engineering & Selection – Identify key features and preprocess the data.
      2. Algorithm Selection – Choose between Decision Tree or Random Forest.
      3. Training & Hyperparameter Tuning – Optimize model parameters for better accuracy.
      4. Evaluation – Assess performance using accuracy, confusion matrix, and other relevant metrics.
             
    - **Model 2:**
      1. Text Preprocessing – Perform tokenization, padding, and word embedding.
      2. Model Architecture Design – Construct an LSTM or GRU model with appropriate layers and parameters.
      3. Training & Hyperparameter Optimization – Train the model and adjust parameters such as learning rate, batch size, and dropout.
      4. Evaluation – Measure performance using accuracy and loss metrics, and possibly other text classification benchmarks (e.g., F1-score).
    
    ## 4. Model Deployment
          - Saving the Models:
             - Machine Learning models are saved as .pkl files.
             - Neural Network models are saved as .h5 files.
          
          - Deployment Using Streamlit:
             - A web-based interactive interface is developed using Streamlit to allow users to input data and receive predictions from both models.
             - The frontend provides a user-friendly way to test and visualize the model’s predictions in real-time.


            ## เเหล่งที่ มา gpt
    """)

# About Model 1 (Machine Learning)
elif page == "📚 About Model 1":
    st.markdown("<h1 style='color: #3498db;'>About Model 1 - Machine Learning</h1>", unsafe_allow_html=True)
    st.write("""
    🧠 **Model 1** is built using **traditional Machine Learning algorithms** to predict whether a user will make a purchase based on their personal and behavioral data.
    This model is commonly used in fields like finance, healthcare, and business analytics.
    """)
    st.markdown("### 🚀 Features:")
    st.write("- Uses structured numerical and categorical data as input.")
    st.write("- Performs **classification** to predict purchase behavior.")
    st.write("- Trained on user attributes such as **Age, Education Level, Income, and Website Visits**.")
    
    st.markdown("### 📊 Dataset Information:")
    st.write("- **Columns:** Age, Education_Level, Income, Website_Visits, Purchase")
    st.write("- **Target Variable:** Purchase (1 = Made a purchase, 0 = Did not make a purchase)")
    st.write("- **Missing Data:** Some missing values in Website_Visits")

# About Model 2 (Neural Network)
elif page == "📚 About Model 2":
    st.markdown("<h1 style='color: #9b59b6;'>About Model 2 - Neural Network for Text</h1>", unsafe_allow_html=True)
    st.write("""
    🔥 **Model 2** is a deep learning model designed to analyze and classify textual data. It uses **Natural Language Processing (NLP)** 
    techniques to predict whether a tweet is related to a disaster.
    """)
    st.markdown("### ✨ Features:")
    st.write("- Takes tweet text as input and converts it into numerical embeddings.")
    st.write("- Uses **Neural Networks (LSTM, GRU, or Dense layers)** for text classification.")
    st.write("- Ideal for **disaster detection in social media posts**.")
    
    st.markdown("### 📊 Dataset Information:")
    st.write("- **Columns:** id, keyword, location, text, target")
    st.write("- **Target Variable:** target (1 = Disaster-related tweet, 0 = Not disaster-related)")
    st.write("- **Missing Data:** Some missing values in location")


# Model 1 - Numeric Prediction
elif page == "📊 Model 1 Machine Learning - Purchase Prediction":
    st.markdown("<h1 style='color: #2ecc71;'>Machine Learning - Purchase Prediction</h1>", unsafe_allow_html=True)
    input_data = st.text_input("🔢 Enter Age, Education_Level, Income, Website_Visits, Purchase (50.0,50000.0,19.0):")
    
    if st.button("⚡ Predict"):
        try:
            data = np.array(input_data.split(","), dtype=float).reshape(1, -1)
            prediction = model1.predict(data)
            st.success(f"🎯 Prediction Result: **{prediction[0]}**")
        except:
            st.error("❌ Invalid input. Please enter numerical values correctly.")

# Model 2 - Text Prediction
elif page == "📝 Model 2 Neural Network - Disaster Tweet Classification":
    st.markdown("<h1 style='color: #e74c3c;'>Neural Network - Disaster Tweet Classification (Text)</h1>", unsafe_allow_html=True)
    input_text = st.text_area("🖊️ Enter text for prediction:")
    
    if st.button("⚡ Predict"):
        try:
            sequences = tokenizer2.texts_to_sequences([input_text])
            padded_sequences = pad_sequences(sequences, maxlen=100)
            prediction = model2.predict(padded_sequences)
            prediction_class = (prediction[0] >= 0.5).astype(int)
            st.success(f"🎯 Prediction Result: **{prediction_class[0]}** (0 = Negative, 1 = Positive)")
        except Exception as e:
            st.error(f"❌ Error: {e}")
