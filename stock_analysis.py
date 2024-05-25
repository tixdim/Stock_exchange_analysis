from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
import sklearn
import joblib
import os

# отключение всех выводов от tensorflow, кроме ошибок
# Значения для TF_CPP_MIN_LOG_LEVEL:
# '0' — все сообщения (по умолчанию).
# '1' — скрыть информационные сообщения.
# '2' — скрыть предупреждения.
# '3' — скрыть все, кроме ошибок
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

app = Flask(__name__)

companies = ["AAPL", "GOOG", "INTC", "NVDA", "TSLA", "AMD", "AMZN", "F", "NFLX", "AAL"]

# количество данных, на которых строится предикт
window_size = 20

# количество образцов данных, обрабатываемых моделью за одну итерацию (лучше поменьше, тк данных не много)
batch_size = 3

# используется для нормализации фичей (чтобы более качественнее обучалась модель)
scaler = sklearn.preprocessing.StandardScaler()


def print_red(text):
    print(f"\033[91m{text}\033[0m")


def print_green(text):
    print(f"\033[92m{text}\033[0m")

def print_yellow(text):
    print(f"\033[93m{text}\033[0m")


def load_data(company):
    print_yellow(f"[INFO]: Loading data for {company}")
    res = yf.Ticker(company)
    df = res.history(period="2y", interval="1h")
    df.to_csv(f"all_data/data_{company}.csv", encoding='utf-8', index=False, columns=["Open", "High", "Low", "Close", "Volume"])
    print_green(f"[INFO]: Successfully loaded data for {company}")


def load_last_data(company):
    res = yf.Ticker(company)
    df = res.history(period="1y", interval="1h").iloc[-window_size:]
    return df


def make_dataset(df, use_scaler=True, shuffle=True):
    # обрезаем датафрейм, т. к. для него не хватит фичей
    features = df[["Close"]].iloc[:-window_size]

    # нормализуем датафрейм на уже настроенном scaler
    if use_scaler:
        features = scaler.transform(features)
    data = np.array(features, dtype=np.float32)

    # создание датасета для временного ряда
    # (нужно, тк модель должна учитывать временные зависимости между данными)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data, targets=df["Close"].iloc[window_size:],
        sequence_length=window_size, sequence_stride=1,
        shuffle=shuffle, batch_size=batch_size
    )
    return ds


def complite_and_fit(model, train_ds, val_ds, num_epochs=20):
    # настройка нейросети
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    # само обучение
    model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, verbose=0)


def train_company(company):
    print_yellow(f"[INFO]: Trying to train {company}")

    df = pd.read_csv(f"all_data/data_{company}.csv")
    print_yellow(f"[INFO]: Size of data {company} - {df.shape}")

    train_size = int(df.shape[0] * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    # настройка scaler под текущую акцию
    scaler.fit(train_df[["Close"]])

    # создаем наши датасеты
    train_ds = make_dataset(df=train_df, use_scaler=True, shuffle=True)
    val_ds = make_dataset(df=val_df, use_scaler=True, shuffle=False)

    # задаем структуру модели
    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(48, return_sequences=False, kernel_regularizer=l2(0.001)),
        Dropout(0.2),
        tf.keras.layers.Dense(1)
    ])

    # обучаем модель
    complite_and_fit(lstm_model, train_ds, val_ds, 30)

    # вывод результатов обучения (loss, mean_absolute_error)
    lstm_model.evaluate(train_ds)
    lstm_model.evaluate(val_ds)

    # сохраняем
    joblib.dump(lstm_model, f'all_models/model_{company}.pkl')

    print_green(f"[INFO]: Successfully trained model for {company}")


def download_all_data(flag_download=True):
    if not flag_download:
        return

    print_yellow(f"[INFO]: Downloading all data")

    if not os.path.isdir("all_data"):
        os.mkdir("all_data")

    for company in companies:
        load_data(company)

    print_green(f"[INFO]: Successfully loaded all data")


def training_all_models(flag_train=True):
    if not flag_train:
        return

    print_yellow(f"[INFO]: Training all data")

    print()

    if not os.path.isdir("all_models"):
        os.mkdir("all_models")

    for company in companies:
        try:
            train_company(company)
        except Exception as e:
            print_red(f"[INFO]: Failed to train model for {company}")
            print_red(f"[INFO]: Try running the model training again!")
            print_red(f"[ERROR]: {e}")
            return

    print_green(f"[INFO]: Successfully trained all data")


if __name__ == "__main__":
    print("[INFO]: Start programme")
    try:
        download_all_data(flag_download=False)
    except:
        print_red("Opps! Something went wrong. Try again pls...")

    try:
        training_all_models(flag_train=True)
    except:
        print_red("Opps! Something went wrong. Try again pls...")