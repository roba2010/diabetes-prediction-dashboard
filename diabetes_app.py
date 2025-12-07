
# Diabetes Prediction App (Enhanced)

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import pickle
from sklearn.metrics import accuracy_score, confusion_matrix

# ---------- إعداد صفحة ستريملِت ----------
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# ---------- تحميل البيانات ----------
df = pd.read_csv("diabetes.csv")

# ---------- تحميل النموذج المحفوظ ----------
with open("diabetes_model.pkl", "rb") as f:
    data = pickle.load(f)
    model = data["model"]
    scaler = data["scaler"]
    feature_cols = data["columns"]

# ---------- الشريط الجانبي ----------
option = st.sidebar.selectbox("Pick a choice:", ["Home", "EDA", "ML"])

#                               HOME
if option == "Home":
    st.title(" Diabetes Prediction App")
    st.markdown("###  Author: **Roba Mohamad**")  # هنا كتبت اسمي  
    st.write(
        """
        This dashboard analyzes **diabetes data** from Kaggle and uses a 
        **Machine Learning model** to predict whether a patient is likely to have diabetes.
        """
    )

    st.markdown("####  Sample of the dataset:")
    st.dataframe(df.head())

    st.markdown(
        """
        **Columns:**
        - `Pregnancies`: عدد مرات الحمل  
        - `Glucose`: مستوى السكر في الدم  
        - `BloodPressure`: ضغط الدم الانبساطي  
        - `SkinThickness`: سماكة ثنايا الجلد  
        - `Insulin`: مستوى الإنسولين  
        - `BMI`: مؤشر كتلة الجسم  
        - `DiabetesPedigreeFunction`: مؤشر يعتمد على التاريخ العائلي  
        - `Age`: عمر المريض  
        - `Outcome`: 0 = لا يعاني من السكري، 1 = يعاني من السكري
        """
    )

#                               EDA
elif option == "EDA":
    st.title(" Exploratory Data Analysis (EDA)")

    # -------- إحصائيات وصفية عامة --------
    st.markdown("### Basic statistics for the dataset:")
    st.write(df.describe().round(2))

    # -------- فلتر بالعمر --------
    st.markdown("### Filter data by Age")

    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())

    age_range = st.slider(
        "Select age range:",
        min_value=min_age,
        max_value=max_age,
        value=(min_age, max_age)
    )

    # فلترة البيانات حسب مدى العمر المختار
    filtered_df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

    st.write(f"Number of records in selected age range: **{len(filtered_df)}**")

    col1, col2 = st.columns(2)

    # -------- توزيع Outcome --------
    with col1:
        st.subheader("Outcome distribution (0 = No Diabetes, 1 = Diabetes)")
        outcome_counts = filtered_df["Outcome"].value_counts().reset_index()
        outcome_counts.columns = ["Outcome", "Count"]
        fig = px.bar(
            outcome_counts,
            x="Outcome",
            y="Count",
            text="Count",
            title="Outcome distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    # -------- توزيع الأعمار --------
    with col2:
        st.subheader("Age distribution")
        fig2 = px.histogram(
            filtered_df,
            x="Age",
            nbins=30,
            title="Age Histogram"
        )
        st.plotly_chart(fig2, use_container_width=True)

    # -------- علاقة Glucose vs BMI --------
    st.subheader("Glucose vs BMI (colored by Outcome)")
    fig3 = px.scatter(
        filtered_df,
        x="Glucose",
        y="BMI",
        color="Outcome",
        title="Glucose vs BMI by Outcome",
        opacity=0.7
    )
    st.plotly_chart(fig3, use_container_width=True)

    # -------- Correlation Heatmap --------
    st.subheader("Correlation Heatmap")
    corr = filtered_df.corr(numeric_only=True)
    fig_corr = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation between numerical features"
    )
    st.plotly_chart(fig_corr, use_container_width=True)

#                               ML
elif option == "ML":
    st.title("🤖 Diabetes Prediction Model")

    # -------- تقييم النموذج على كامل البيانات (للعرض بس) --------
    X_all = df.drop("Outcome", axis=1)
    y_all = df["Outcome"]
    X_all_scaled = scaler.transform(X_all)
    y_pred_all = model.predict(X_all_scaled)

    acc_all = accuracy_score(y_all, y_pred_all)
    cm = confusion_matrix(y_all, y_pred_all)

    st.markdown(f"**Model accuracy on full dataset:** `{acc_all:.2%}`")

    st.markdown("**Confusion Matrix:**")
    st.write(pd.DataFrame(cm,
                          index=["True 0 (No Diabetes)", "True 1 (Diabetes)"],
                          columns=["Pred 0", "Pred 1"]))

    st.markdown("---")
    st.write("Enter the patient data below and click **Predict**:")

    # -------- إدخالات المستخدم --------
    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("BloodPressure", min_value=0, max_value=200, value=70)
        skin_thickness = st.number_input("SkinThickness", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
        dpf = st.number_input("DiabetesPedigreeFunction", min_value=0.0, max_value=3.0, value=0.5)
        age = st.number_input("Age", min_value=1, max_value=120, value=30)

    btn = st.button("Predict")

    if btn:
        # ترتيب المدخلات مثل ترتيب الأعمدة
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])

        # نفس الـ scaler المستخدم في التدريب
        input_scaled = scaler.transform(input_data)

        # احتمال الإصابة (الفئة 1)
        proba = model.predict_proba(input_scaled)[0][1]
        result = model.predict(input_scaled)[0]

        st.markdown(f"**Predicted probability of diabetes:** `{proba*100:.1f}%`")

        if result == 1:
            st.error("⚠ The model predicts that the patient is **LIKELY to have diabetes**.")
        else:

            st.success(" The model predicts that the patient is **NOT likely to have diabetes**.")
            # -----------------------------------------
# قسم جديد لعرض الرسوم البيانية بالـ Plotly
# -----------------------------------------

st.markdown("---")
st.header("📈 Data Visualization")

st.write("استخدمي هذه الرسوم لاستكشاف بيانات السكري بصريًا.")

# 1) توزيع أي عمود (Histogram)
st.subheader("Histogram – توزيع أحد المتغيرات")
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

selected_col = st.selectbox(
    "اختر العمود الذي تريدين مشاهدة توزيعه:",
    numeric_cols
)

fig_hist = px.histogram(
    df,
    x=selected_col,
    nbins=30,
    title=f"Distribution of {selected_col}",
    marginal="box"
)
st.plotly_chart(fig_hist, use_container_width=True)

# 2) العلاقة بين متغيرين (Scatter plot)
st.subheader("Scatter plot – العلاقة بين متغيرين")

x_var = st.selectbox("المتغير على المحور السيني (X):", numeric_cols, key="x_var")
y_var = st.selectbox("المتغير على المحور الصادي (Y):", numeric_cols, key="y_var")

color_col = "Outcome" if "Outcome" in df.columns else None

fig_scatter = px.scatter(
    df,
    x=x_var,
    y=y_var,
    color=color_col,
    title=f"Relationship between {x_var} and {y_var}",
    trendline="ols"
)
st.plotly_chart(fig_scatter, use_container_width=True)
