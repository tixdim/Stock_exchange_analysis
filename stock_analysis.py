from load_and_train import *
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

companies = ["AAPL", "GOOG", "INTC", "NVDA", "TSLA", "AMD", "AMZN", "F", "NFLX", "AAL"]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction, error, selected_company, latest_dates, latest_prices = [None] * 5

    if request.method == 'POST':
        company = request.form['company']
        selected_company = company

        if company not in companies:
            error = "Company not found"

        else:
            # загрузка последних данных
            df = load_last_data(company)
            latest_data = df["Close"].values[:].reshape(-1, 1)
            latest_dates = df.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
            latest_prices = df["Close"].values.tolist()

            # настройка scaler под текущую акцию
            try:
                df_all = pd.read_csv(f"all_data/data_{company}.csv")
                train_df = df_all.iloc[:int(df_all.shape[0] * 0.8)]

                scaler = sklearn.preprocessing.StandardScaler()
                scaler.fit(train_df[["Close"]])

                scaled_latest_data = scaler.transform(latest_data)
                formatted_data = np.array(scaled_latest_data, dtype=np.float32).reshape(1, window_size, 1)

                # загрузка модели и сам предикт
                try:
                    model = joblib.load(f'all_models/model_{company}.pkl')
                    predicted_price = model.predict(formatted_data)
                    prediction = round(predicted_price[0][0], 3)

                except Exception as e:
                    print_red(f"[INFO]: Not find model for company {company}")
                    print_red(f"[ERROR]: {e}")
                    error = f"Not find model for company {company}"

            except Exception as e:
                print_red(f"[INFO]: Not find data for company {company}")
                print_red(f"[ERROR]: {e}")
                error = f"Not find data for company {company}"

    return render_template('index.html', prediction=prediction, error=error, companies=companies, selected_company=selected_company, latest_dates=latest_dates, latest_prices=latest_prices)

if __name__ == '__main__':
    app.run(debug=True)
