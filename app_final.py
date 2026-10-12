from flask import Flask, render_template, request
import joblib
import pandas as pd


app = Flask(__name__)
import os

print("Static Folder Path:", os.path.abspath("static/Far.jpg"))
print("File Exists:", os.path.exists("static/Far.jpg"))


@app.route("/")
def home():
    return render_template("home2.html")  # The first page

@app.route("/index")
def index():
    return render_template("index_final.html")  # The prediction form

@app.route("/about_us")
def about_us():
    return render_template("About_us.html")  # About Us page

@app.route("/predict", methods=["POST"])
def predict():
    # Fetch form data
    district = request.form["district"]
    market = request.form["market"]
    commodity = request.form["commodity"]
    variety = request.form["variety"]
    grade = request.form["grade"]
    season = request.form["season"]
    temperature = request.form["temperature"]
    rainfall = request.form["rainfall"]
    year = request.form["year"]
    month = request.form["month"]
    day = request.form["day"]

    # Convert input to DataFrame
    input_data = pd.DataFrame([[district, market, commodity, variety, grade, season,
                                temperature,rainfall, year, month, day]],
                              columns=['District Name', 'Market Name', 'Commodity', 'Variety', 'Grade','Season', 
                                       'Temperature (°C)',  'Rainfall (mm)', 
                                       'Year', 'Month', 'Day'])

    # Load the model and predict
    loaded_pipeline = joblib.load('Final_model_kharif.pkl')
    prediction = loaded_pipeline.predict(input_data)[0]

    prediction_details = [
        ("District", district),
        ("Market", market),
        ("Commodity", commodity),
        ("Variety", variety),
        ("Grade", grade),
        ("Season", season),
        ("Temperature", f"{temperature} °C"),
        ("Rainfall", f"{rainfall} mm"),
        ("Date", f"{day}/{month}/{year}"),
    ]

    return render_template(
        "predict.html",
        predicted_price=prediction,
        prediction_details=prediction_details,
    )

if __name__ == "__main__":
    app.run(debug=True)
