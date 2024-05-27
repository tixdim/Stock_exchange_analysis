import os
# отключение всех выводов от tensorflow, кроме ошибок
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import sklearn
import joblib


companies = ["AAPL", "GOOG", "INTC", "NVDA", "TSLA", "AMD", "AMZN", "F", "NFLX", "AAL"]

# количество данных, на которых строится предикт
window_size = 20

# количество образцов данных, обрабатываемых моделью за одну итерацию (лучше поменьше, тк данных не много)
batch_size = 3

# используется для нормализации фичей (чтобы более качественнее обучалась модель)
scaler = None


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
    print_green(f"[INFO]: Successfully loaded ans saved data for {company}")


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

    # обнволение scaler для текущей компании
    global scaler
    scaler = sklearn.preprocessing.StandardScaler()

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
        tf.keras.layers.LSTM(48, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])

    # обучаем модель
    complite_and_fit(lstm_model, train_ds, val_ds, 30)

    # вывод результатов обучения (loss, mean_absolute_error)
    lstm_model.evaluate(train_ds)
    lstm_model.evaluate(val_ds)

    # сохраняем
    joblib.dump(lstm_model, f'all_models/model_{company}.pkl')

    print_green(f"[INFO]: Successfully trained ans saved model for {company}")
    print()


def download_all_data(flag_download=True):
    if not flag_download:
        return

    print_yellow(f"[INFO]: Downloading all data")
    print()

    if not os.path.isdir("all_data"):
        os.mkdir("all_data")

    for company in companies:
        load_data(company)

    print()
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
    print()

    flag_download = False
    flag_train = False

    try:
        download_all_data(flag_download=flag_download)
    except Exception as e:
        print_red("Opps! Something went wrong. Try again pls...")
        print_red(f"[ERROR]: {e}")

    if flag_download:
        print()

    try:
        training_all_models(flag_train=flag_train)
    except Exception as e:
        print_red("Opps! Something went wrong. Try again pls...")
        print_red(f"[ERROR]: {e}")

    if flag_train:
        print()

    print("[INFO]: Finish programme")