from flask import Flask, render_template, request
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import sklearn
import joblib

app = Flask(__name__)


def load_last_data(company, window_size):
    res = yf.Ticker(company)
    df = res.history(period="1y", interval="1h").iloc[-window_size:]
    return df


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        company = request.form['company']
        window_size = 20
        if company not in companies:
            print("gavno")

        df = load_last_data(company, window_size)
        latest_data = df["Close"].values[:].reshape(-1, 1)

        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(latest_data)
        scaled_latest_data = scaler.transform(latest_data)
        formatted_data = np.array(scaled_latest_data, dtype=np.float32).reshape(1, window_size, 1)

        model = joblib.load(f'all_models/model_{company}.pkl')

        predicted_price = model.predict(formatted_data)
        prediction = predicted_price[0][0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)