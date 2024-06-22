import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Чтение данных из CSV файла
data = pd.read_csv('spam.csv')

# Преобразование категорий в числовые значения (1 - спам, 0 - не спам)
data['label'] = data['category'].map({'spam': 1, 'ham': 0})

# Получение текстов и меток
texts = data['text'].values
labels = data['label'].values

# Преобразование текстов в векторы признаков
max_words = 1000
max_len = 100

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
X = tokenizer.texts_to_sequences(texts)
X = pad_sequences(X, maxlen=max_len)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)

# Создание модели нейронной сети
model = Sequential()
model.add(Embedding(max_words, 32, input_length=max_len))
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
start_time = time.time()
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_split=0.1, verbose=2)
training_time = time.time() - start_time

# Оценка модели на тестовой выборке

y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int).reshape(-1)


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

user_input = ""
while (user_input != "stop"):
    # Ввод произвольного текста для проверки на спам
    user_input = input("Введите текст для проверки на спам: ")
    # Преобразование ввода пользователя в вектор признаков
    user_input_seq = tokenizer.texts_to_sequences([user_input])
    user_input_pad = pad_sequences(user_input_seq, maxlen=max_len)
    # Предсказание на основе ввода пользователя
    start_time = time.time()
    user_prediction_prob = model.predict(user_input_pad)
    user_prediction = (user_prediction_prob > 0.5).astype(int)[0][0]
    prediction_time = time.time() - start_time
    print(f"Prediction time: {prediction_time:.4f} seconds")
    # Вывод результата
    if user_prediction == 1:
        print("Текст является спамом.")
    else:
        print("Текст не является спамом.")
