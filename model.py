import yfinance as yf
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler

# close - [i > i - 1] : (0, 1)
def load_data(company):
    res = yf.Ticker(company)
    # добавить больше дней
    df = res.history(period="90d", interval="1h")
    print(df.shape)
    df.to_csv(f"data_{company}.csv", encoding='utf-8', index=False, columns=["Open", "High", "Low", "Close", "Volume"])


def make_dataset(df, scaler, window_size, batch_size, use_scaler=True, shuffle=True):
    features = df[["Close"]].iloc[:-window_size]
    if use_scaler:
        features = scaler.transform(features)
    data = np.array(features, dtype=np.float32)
    ds = tf.keras.preprocessing.timeseries_dataset_from_array(
        data=data, targets=df["Close"].iloc[window_size:],
        sequence_length=window_size, sequence_stride=1,
        shuffle=shuffle, batch_size=batch_size
    )
    return ds


def complite_and_fit(model, train_ds, val_ds, num_epochs=40):
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanSquaredError()])

    history = model.fit(train_ds, epochs=num_epochs, validation_data=val_ds, verbose=0)
    return history


if __name__ == "__main__":
    if not os.path.exists("data_AAPL.csv"):
        load_data("AAPL")

    df = pd.read_csv("data_AAPL.csv")
    print(df.shape)

    train_size = int(df.shape[0] * 0.8)
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]

    print(train_df.shape, val_df.shape)

    scaler = StandardScaler()
    scaler.fit(train_df[["Close"]])
    window_size, batch_size = 30, 20
    train_ds = make_dataset(df=train_df, scaler=scaler, window_size=window_size, batch_size=batch_size,
                            use_scaler=True, shuffle=True)
    val_ds = make_dataset(df=val_df, scaler=scaler, window_size=window_size, batch_size=batch_size,
                          use_scaler=True, shuffle=True)

    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(64, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    print(123)
    history = complite_and_fit(lstm_model, train_ds, val_ds, num_epochs=40)
    lstm_model.evaluate(train_ds)
    lstm_model.evaluate(val_ds)