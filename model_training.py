import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Загрузка данных
df = pd.read_csv('data/WineQT.csv').drop('Id', axis=1)
print("Class distribution:")
print(df['quality'].value_counts().sort_index())

# Объединим редкие классы (3, 4 и 8,9) для балансировки
df['quality'] = df['quality'].apply(lambda x: 5 if x == 3 or x == 4 else (7 if x == 8 or x == 9 else x))
print("\nClass distribution after merging rare classes:")
print(df['quality'].value_counts().sort_index())

# Разделение данных
X = df.drop('quality', axis=1)
y = df['quality']

# Стратифицированное разделение
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Проверяем количество образцов в каждом классе обучающей выборки
print("\nTraining set class distribution:")
print(y_train.value_counts().sort_index())

# Применение SMOTE для балансировки классов с безопасным k_neighbors
try:
    # Автоматически определяем безопасное k_neighbors
    min_samples = min(y_train.value_counts())
    safe_k = min(5, min_samples - 1)  # k_neighbors должно быть <= min_samples-1

    print(f"\nApplying SMOTE with k_neighbors={safe_k}")
    smote = SMOTE(random_state=42, k_neighbors=safe_k)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    print("\nAfter SMOTE:")
    print(pd.Series(y_train_res).value_counts().sort_index())
except Exception as e:
    print(f"SMOTE failed: {e}")
    print("Using original training data without SMOTE")
    X_train_res, y_train_res = X_train, y_train

# Масштабирование
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Сохранение scaler
joblib.dump(scaler, 'scaler.pkl')
print("\nScaler saved as 'scaler.pkl'")

# Baseline модели
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced')
}

# Обучение и оценка
best_model = None
best_f1 = 0

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train_res)
    y_pred = model.predict(X_val_scaled)

    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred, average='weighted', zero_division=0)

    print(f"{name}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print(classification_report(y_val, y_pred, zero_division=0))

    # Сохранение лучшей модели
    if f1 > best_f1:
        best_f1 = f1
        best_model = model
        print("New best model!")

# Сохранение лучшей модели
if best_model:
    joblib.dump(best_model, 'best_model.pkl')
    print(f"\nSaved best model ({type(best_model).__name__}) with F1-score: {best_f1:.4f}")

# Оценка на тестовом наборе
if best_model:
    test_pred = best_model.predict(X_test_scaled)
    test_accuracy = accuracy_score(y_test, test_pred)
    test_f1 = f1_score(y_test, test_pred, average='weighted', zero_division=0)
    print(f"\nFinal Test Results:")
    print(f"Accuracy: {test_accuracy:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(classification_report(y_test, test_pred, zero_division=0))