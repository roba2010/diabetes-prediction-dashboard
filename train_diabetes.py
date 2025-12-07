#تدريب النموذج logistic ,random
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.impute import SimpleImputer
import numpy as np
import pickle


# قراءة بيانات السكري من ملف CSV في DataFrame
df = pd.read_csv("diabetes.csv")

# ------ 2) Replace zeros with NaN in specific columns ------
# في بعض الأعمدة، القيمة 0 غير منطقية (يعني في الحقيقة Missing)
# لذلك نستبدل 0 بقيمة NaN (قيمة مفقودة) لتصحيح البيانات
cols_with_zero_invalid = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df[cols_with_zero_invalid] = df[cols_with_zero_invalid].replace(0, np.nan)

# ------ 3) Impute missing values ------
# الآن بعد أن وضعنا NaN، نحتاج لتعويض القيم المفقودة
# SimpleImputer(strategy="median") يعوّض كل NaN بالقيمة "الوسيط" لكل عمود
imputer = SimpleImputer(strategy="median")
df[cols_with_zero_invalid] = imputer.fit_transform(df[cols_with_zero_invalid])

# طباعة عدد القيم المفقودة بعد التصحيح (يجب أن تصبح كلها 0)
print("Missing values after fixing:\n", df.isna().sum())

# ------ 4) Split ------
# فصل المتغيرات المفسرة X عن العمود الهدف y (Outcome: 0 أو 1)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ------ 5) Scale ------
# عمل Scaling للميزات باستخدام MinMaxScaler
# هذا يحول كل عمود لقيم بين 0 و 1 
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# ------ 6) Train-test split ------
# تقسيم البيانات إلى جزء للتدريب وجزء للاختبار
# test_size=0.2 يعني 20% للـ test و 80% للـ train
# stratify=y يحافظ على نفس نسبة المصابين/غير المصابين في كلا المجموعتين
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ------ 7) Logistic Regression ------
# إنشاء نموذج الانحدار اللوجستي وتدريبه على بيانات التدريب
log_clf = LogisticRegression(max_iter=2000)
log_clf.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار
y_pred_log = log_clf.predict(X_test)

# حساب الدقة
acc_log = accuracy_score(y_test, y_pred_log)

print("\n===== Logistic Regression =====")
print("Accuracy:", acc_log)
print(confusion_matrix(y_test, y_pred_log))
print(classification_report(y_test, y_pred_log))

# ------ 8) Random Forest ------
# إنشاء نموذج الغابة العشوائية RandomForestClassifier
# n_estimators=400 يعني 400 شجرة
# class_weight='balanced' لموازنة الفئات (مصاب/غير مصاب)
rf_clf = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    class_weight="balanced",
    min_samples_split=3,
    min_samples_leaf=1,
)

# تدريب نموذج Random Forest
rf_clf.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار
y_pred_rf = rf_clf.predict(X_test)

# حساب الدقة
acc_rf = accuracy_score(y_test, y_pred_rf)

print("\n===== Random Forest =====")
print("Accuracy:", acc_rf)
print(confusion_matrix(y_test, y_pred_rf))
print(classification_report(y_test, y_pred_rf))

# ------ 9) Choose best model ------
# اختيار أفضل نموذج بناءً على الـ Accuracy
# إذا كانت دقة Random Forest أكبر أو تساوي Logistic → نختاره، وإلا نختار Logistic
best_model = rf_clf if acc_rf >= acc_log else log_clf
best_name = "RandomForestClassifier" if acc_rf >= acc_log else "LogisticRegression"

print("\n>>> Best model selected:", best_name)

# ------ 10) Save model ------
# حفظ أفضل نموذج + الـ scaler + أسماء الأعمدة داخل ملف pkl
model_data = {
    "model": best_model,      # النموذج الأفضل
    "scaler": scaler,         # الـ MinMaxScaler المستخدم في التدريب
    "columns": list(X.columns),  # أسماء الأعمدة (الخصائص)
}

with open("diabetes_model.pkl", "wb") as f:
    pickle.dump(model_data, f)

print("\n Best model saved to diabetes_model.pkl")