import streamlit as st
import pickle
import numpy as np

# โหลดโมเดลที่เซฟไว้
with open("finalized_model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Machine Learning Model Deployment")

# รับค่าจากผู้ใช้
input_data = st.text_input("กรอกค่าตัวเลขคั่นด้วย , เช่น 5.1, 3.5, 1.4, 0.2")
if st.button("Predict"):
    try:
        data = np.array(input_data.split(","), dtype=float).reshape(1, -1)
        prediction = model.predict(data)
        st.write(f"ผลลัพธ์การพยากรณ์: {prediction[0]}")
    except:
        st.error("กรุณากรอกข้อมูลให้ถูกต้อง")
