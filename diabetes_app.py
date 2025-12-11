# -------------------- IMPORTS --------------------
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Page Config
st.set_page_config(
    page_title="Diabetes Prediction Dashboard",
    layout="wide",
    page_icon="🩺"
)

# Load Data
df = pd.read_csv("diabetes.csv")

# Load Model
model_data = pickle.load(open("diabetes_model.pkl", "rb"))
model = model_data["model"]
scaler = model_data["scaler"]

# Sidebar
page = st.sidebar.selectbox("📌 اختر صفحة", ["🏠 Home", "📊 EDA", "🤖 ML Prediction"])


# ============================================================
# 🏠 HOME PAGE
# ============================================================
if page == "🏠 Home":
    st.title("🩺 Diabetes Prediction Dashboard")
    st.markdown("### **Author: Roba Mohamad**")
    st.write("هذا المشروع يقوم بتحليل بيانات مرضى السكري وبناء نموذج للتنبؤ بإصابة المريض بالسكري.")

    st.subheader("📌 نظرة عامة على البيانات")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("عدد المرضى في البيانات", df.shape[0])
    with col2:
        st.metric("عدد الخصائص (Features)", df.shape[1] - 1)
    with col3:
        st.metric("عدد المصابين بالسكري", df["Outcome"].sum())

    col4, col5 = st.columns(2)
    with col4:
        st.metric("عدد غير المصابين", (df["Outcome"] == 0).sum())
    with col5:
        st.metric("نسبة الإصابة بالسكري", f"{df['Outcome'].mean()*100:.2f}%")

    st.write("### عرض أول 5 صفوف من البيانات:")
    st.dataframe(df.head())


# ============================================================
# 📊 EDA PAGE
# ============================================================
elif page == "📊 EDA":
    st.title("📊 Exploratory Data Analysis (EDA)")

    st.subheader("1️⃣ توزيع العمر")
    fig = px.histogram(df, x="Age", nbins=30, color="Outcome",
                       title="Age Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("2️⃣ توزيع مستوى الغلوكوز")
    fig = px.histogram(df, x="Glucose", nbins=30, color="Outcome",
                       title="Glucose Distribution")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("3️⃣ Boxplot لـ BMI حسب الإصابة")
    fig = px.box(df, x="Outcome", y="BMI", color="Outcome",
                 title="BMI by Outcome")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("4️⃣ عدد الحالات المصابة وغير المصابة")
    fig = px.bar(df["Outcome"].value_counts(), title="Outcome Count")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("5️⃣ العلاقة بين متغيرين (Scatter)")
    x_var = st.selectbox("اختر المتغير الأول:", df.columns[:-1])
    y_var = st.selectbox("اختر المتغير الثاني:", df.columns[:-1])

    fig = px.scatter(df, x=x_var, y=y_var, color="Outcome",
                     title=f"{x_var} vs {y_var}")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("6️⃣ Pairplot لعينة من البيانات")
    st.write("هذا الرسم يعطي فكرة عن العلاقات بين عدة متغيرات.")
    sample_df = df.sample(200)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.pairplot(sample_df.iloc[:, :5])
    st.pyplot(fig)


# ============================================================
# 🤖 ML Prediction
# ============================================================
elif page == "🤖 ML Prediction":
    st.title("🤖 Diabetes Prediction Model")

    st.write("أدخل بيانات المريض ثم اضغط **Predict**")

    col1, col2, col3 = st.columns(3)

    with col1:
        Pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        Glucose = st.number_input("Glucose", 0, 300, 120)
        BloodPressure = st.number_input("BloodPressure", 0, 200, 70)

    with col2:
        SkinThickness = st.number_input("SkinThickness", 0, 100, 20)
        Insulin = st.number_input("Insulin", 0, 900, 80)
        BMI = st.number_input("BMI", 0.0, 70.0, 25.0)

    with col3:
        DPF = st.number_input("DiabetesPedigreeFunction", 0.0, 3.0, 0.5)
        Age = st.number_input("Age", 18, 90, 30)

    if st.button("🔮 Predict"):
        user_input = [[Pregnancies, Glucose, BloodPressure, SkinThickness,
                       Insulin, BMI, DPF, Age]]

        scaled_input = scaler.transform(user_input)
        prediction = model.predict(scaled_input)
        prob = model.predict_proba(scaled_input)[0][1]

        st.write(f"### 🔢 احتمال الإصابة: **{prob*100:.2f}%**")

        if prediction == 1:
            st.error("🛑 النموذج يتوقع أن المريض **مصاب بالسكري**.")
        else:
            st.success("✅ النموذج يتوقع أن المريض **غير مصاب بالسكري**.")
