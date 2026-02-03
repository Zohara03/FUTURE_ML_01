from flask import Flask, render_template, jsonify, request
import pandas as pd
from prophet import Prophet
import os

app = Flask(__name__)

def build_forecast(periods=90):
    if not os.path.exists("sales_data.csv"):
        raise FileNotFoundError("sales_data.csv not found in project folder")

    # Load dataset
    df = pd.read_csv("sales_data.csv")

    # Parse Datetime in day-month-year hour:minute format
    df["Datetime"] = pd.to_datetime(df["Datetime"], format="%d-%m-%Y %H:%M")

    # Rename for Prophet
    df = df.rename(columns={"Datetime": "ds", "Count": "y"})

    # Drop ID if present
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])

    # Handle missing values
    df["y"] = df["y"].fillna(method="ffill")

    # Build Prophet model
    model = Prophet(yearly_seasonality=True, seasonality_mode="multiplicative")
    model.fit(df)

    # Forecast future periods (periods = number of hours here)
    future = model.make_future_dataframe(periods=periods, freq="H")
    forecast = model.predict(future)

    return forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(periods)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/forecast")
def forecast_api():
    try:
        days = int(request.args.get("days", 90))
        forecast = build_forecast(periods=days)
        result = [
            {
                "date": str(row["ds"]),
                "forecast": float(row["yhat"]),
                "lower_bound": float(row["yhat_lower"]),
                "upper_bound": float(row["yhat_upper"])
            }
            for _, row in forecast.iterrows()
        ]
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)