import time
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# Чтение данных из CSV файла
data = pd.read_csv('spam.csv') 

# Преобразование категорий в числовые значения (1 - спам, 0 - не спам)
data['label'] = data['category'].map({'spam': 1, 'ham': 0})

# Получение текстов и меток
texts = data['text'].values
labels = data['label'].values

# Преобразование текстов в векторы признаков
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Обучение модели k-ближайших соседей
k = 20  # Выбор числа соседей
model = KNeighborsClassifier(n_neighbors=k)

start_time = time.time()
model.fit(X_train, y_train)
training_time = time.time() - start_time

# Предсказание на тестовой выборке
y_pred = model.predict(X_test)

# Вычисление метрик
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# Вывод результатов
print(f"Accuracy: {accuracy:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Precision: {precision:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"Training time: {training_time:.4f} seconds")

# Ввод произвольного текста для проверки на спам
user_input = "" 

while(user_input != "stop"):
    user_input = input("Введите текст для проверки на спам: ")
    # Преобразование ввода пользователя в вектор признаков
    user_input_vector = vectorizer.transform([user_input])
    start_time = time.time()
    # Предсказание на основе ввода пользователя
    user_prediction = model.predict(user_input_vector)[0]
    prediction_time = time.time() - start_time
    print(f"Prediction time: {prediction_time:.4f} seconds")
    # Вывод результата
    if user_prediction == 1:
        print("-----Текст является спамом.-----")
    else:
        print("-----Текст не является спамом.-----")
