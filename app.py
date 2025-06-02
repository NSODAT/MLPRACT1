from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)
model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Загрузка датасета для примеров
df = pd.read_csv('data/WineQT.csv').drop('Id', axis=1)
df['quality'] = df['quality'].apply(lambda x: 5 if x == 3 or x == 4 else (7 if x == 8 or x == 9 else x))

# Названия признаков
feature_names = [
    'Fixed Acidity', 'Volatile Acidity', 'Citric Acid',
    'Residual Sugar', 'Chlorides', 'Free Sulfur Dioxide',
    'Total Sulfur Dioxide', 'Density', 'pH', 'Sulphates', 'Alcohol'
]

# Единицы измерения
units = [
    'г/дм³', 'г/дм³', 'г/дм³',
    'г/дм³', 'г/дм³', 'мг/дм³',
    'мг/дм³', 'г/см³', '', 'г/дм³', '%'
]


@app.route('/')
def home():
    # Выбираем случайный пример из датасета
    sample = df.sample(1)
    sample_features = sample.drop('quality', axis=1).values[0]
    sample_quality = sample['quality'].values[0]

    # Предсказание для примера
    scaled_features = scaler.transform([sample_features])
    prediction = model.predict(scaled_features)[0]

    # Вычисляем точность для примера
    accuracy = 100 - 100 * abs(sample_quality - prediction) / 9

    # Формируем список характеристик для отображения
    features_list = []
    for i in range(len(sample_features)):
        features_list.append({
            'name': feature_names[i],
            'value': sample_features[i],
            'unit': units[i]
        })

    return render_template('index.html',
                           features_list=features_list,
                           sample_quality=sample_quality,
                           prediction=prediction,
                           accuracy=accuracy)
@app.route('/predict-page')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)
    return jsonify({'quality': int(prediction[0])})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)