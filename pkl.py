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
                         "📊 Model 1 (Numeric Prediction)", 
                         "📝 Model 2 (Text Prediction)", 
                         "📖 About Model 1", 
                         "📖 About Model 2"])

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
    ## 1. การเตรียมข้อมูล (Data Preparation)
    - 1.1 ข้อมูลเชิงตัวเลข (Numerical Data) สำหรับ Machine Learning
        - ต้องมีการ ตรวจสอบค่าที่หายไป (Missing Values) และจัดการให้เหมาะสม
        - ทำ Normalization หรือ Standardization เพื่อลดความแตกต่างของช่วงค่าของฟีเจอร์ต่าง ๆ
        - เลือกฟีเจอร์ที่สำคัญ (Feature Selection) และอาจสร้างฟีเจอร์ใหม่ (Feature Engineering)
    
    - 1.2 ข้อมูลข้อความ (Text Data) สำหรับ Neural Network
        - แปลงข้อความเป็นตัวเลขผ่าน Tokenization
        - จัดการความยาวของข้อมูลโดยใช้ Padding หรือ Truncating
        - ใช้ Word Embedding เพื่อสร้างตัวแทนของคำที่มีความหมายใกล้เคียงกันให้อยู่ในรูปเวกเตอร์
    
    ## 2. ทฤษฎีของอัลกอริธึมที่ใช้
    - **Model 1:** อัลกอริธึมสำหรับ Machine Learning
        - สำหรับข้อมูลเชิงตัวเลข โมเดลใช้ Decision Tree หรือ Random Forest
             - Decision Tree: ใช้โครงสร้างต้นไม้ในการแบ่งกลุ่มข้อมูลตามเงื่อนไขต่าง ๆ
             - Random Forest: ใช้ต้นไม้หลายต้นมารวมกันเพื่อลด Overfitting และเพิ่มความแม่นยำ

    - **Model 2:** อัลกอริธึมสำหรับ Neural Network
        - สำหรับข้อมูลข้อความ ใช้โมเดล LSTM หรือ GRU ซึ่งเป็นประเภทของ Recurrent Neural Network (RNN)
             - LSTM (Long Short-Term Memory): ออกแบบมาเพื่อจดจำลำดับของข้อมูลได้นานขึ้น
             - GRU (Gated Recurrent Unit): คล้าย LSTM แต่มีโครงสร้างที่เบากว่า ทำให้ฝึกได้เร็วขึ้น
    
    ## 3. ขั้นตอนการพัฒนาโมเดล
    - **Model 1:**
      1. Feature Engineering และ Feature Selection
      2. เลือกอัลกอริธึม (Decision Tree / Random Forest)
      3. ฝึกโมเดล (Training) และปรับ Hyperparameters
      4. ประเมินผลลัพธ์ด้วย Accuracy และ Confusion Matrix
             
    - **Model 2:**
      1. Preprocessing ข้อมูล (Tokenization, Padding, Word Embedding)
      2. สร้างโมเดล LSTM หรือ GRU
      3. ฝึกโมเดล (Training) และปรับค่า Hyperparameters
      4. ประเมินผลลัพธ์โดยใช้ Accuracy และ Loss Metrics
    
    ## 4. การนำโมเดลไปใช้งาน (Deployment)
    - The trained models are saved as `.pkl` (for ML) and `.h5` (for NN).
    - Streamlit is used as the frontend for easy interaction.
    """)

# About Model 1 (Machine Learning)
elif page == "📖 About Model 1":
    st.markdown("<h1 style='color: #3498db;'>About Model 1 - Machine Learning</h1>", unsafe_allow_html=True)
    st.write("""
    🧠 **Model 1** is built using **traditional Machine Learning algorithms** to predict outcomes based on numerical input data. 
    This model is commonly used in fields like finance, healthcare, and business analytics.
    """)
    st.markdown("### 🚀 Features:")
    st.write("- Uses numerical data as input.")
    st.write("- Can perform **regression** or **classification** tasks.")
    st.write("- Trained using structured datasets.")

# About Model 2 (Neural Network)
elif page == "📖 About Model 2":
    st.markdown("<h1 style='color: #9b59b6;'>About Model 2 - Neural Network for Text</h1>", unsafe_allow_html=True)
    st.write("""
    🔥 **Model 2** is a deep learning model designed to analyze and classify textual data. It uses **Natural Language Processing (NLP)** 
    techniques and is trained with a Tokenizer to convert words into numerical sequences for prediction.
    """)
    st.markdown("### ✨ Features:")
    st.write("- Takes text as input and converts it into word embeddings.")
    st.write("- Uses **Neural Networks (LSTM, GRU, or Dense layers)** for prediction.")
    st.write("- Ideal for **sentiment analysis**, **spam detection**, and **text classification**.")

# Model 1 - Numeric Prediction
elif page == "📊 Model 1 (Numeric Prediction)":
    st.markdown("<h1 style='color: #2ecc71;'>Predict with Machine Learning Model</h1>", unsafe_allow_html=True)
    input_data = st.text_input("🔢 Enter numerical values separated by commas (e.g., 5.1, 3.5, 1.4, 0.2):")
    
    if st.button("⚡ Predict"):
        try:
            data = np.array(input_data.split(","), dtype=float).reshape(1, -1)
            prediction = model1.predict(data)
            st.success(f"🎯 Prediction Result: **{prediction[0]}**")
        except:
            st.error("❌ Invalid input. Please enter numerical values correctly.")

# Model 2 - Text Prediction
elif page == "📝 Model 2 (Text Prediction)":
    st.markdown("<h1 style='color: #e74c3c;'>Predict with Neural Network (Text)</h1>", unsafe_allow_html=True)
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
