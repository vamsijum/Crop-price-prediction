# Crop Price Prediction Web Application

A Flask-based machine learning application that predicts crop market prices using historical agricultural data. The project helps farmers, traders, and agricultural stakeholders estimate crop prices based on district, market, commodity, variety, season, and weather conditions.

## Project Overview

This application uses a trained Random Forest Regressor pipeline to predict the modal price of crops in Indian agricultural markets. It accepts input such as:

- District name
- Market name
- Commodity
- Variety
- Grade
- Season
- Temperature
- Rainfall
- Year, Month, and Day

The model then returns an estimated modal price in Rs./Quintal.

## Application Name

APMandiPredict

## Features

- Crop price prediction using machine learning
- User-friendly web interface built with Flask
- Dynamic district-to-market selections
- Commodity, variety, and grade-based choices
- Seasonal and weather-based prediction inputs
- Real-time prediction result display
- Responsive front-end pages for home, prediction form, and about section

## Technologies Used

- Python
- Flask
- Pandas
- Scikit-learn
- Joblib
- HTML/CSS/JavaScript

## Project Structure

```text
Crop_Price_prediction_PG_Mini_Project/
├── app_final.py                  # Flask application entry point
├── final_model.py                # Model training script
├── Final_model_kharif.pkl        # Trained model artifact
├── final_datset.csv              # Dataset used for model development
├── final_datset_kharif.csv      # Kharif dataset used by the app
├── static/                       # Static assets such as images and CSS
│   ├── Far.jpg
│   ├── second.jpg
│   └── app.css
├── templates/                    # HTML templates for the web app
│   ├── home2.html
│   ├── index_final.html
│   ├── About_us.html
│   └── predict.html
└── README.md                     # Project documentation
```

## How the Prediction Works

1. The user enters crop and market information in the prediction form.
2. The form data is collected in `app_final.py` using Flask request handling.
3. The inputs are converted into a pandas DataFrame with the required feature names.
4. The trained model (`Final_model_kharif.pkl`) is loaded using `joblib`.
5. The model predicts the modal crop price.
6. The result is shown on the web page along with the user-entered details.

## Setup Instructions

### Prerequisites

- Python 3.8 or above
- pip
- Virtual environment (recommended)

### Installation

1. Open a terminal in the project directory.
2. Create and activate a virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install flask pandas scikit-learn joblib
```

### Run the Application

Start the Flask server:

```bash
python app_final.py
```

Then open the browser and visit:

```text
http://127.0.0.1:5000/
```

## Usage

1. Open the homepage.
2. Click on the price prediction page.
3. Select a district and market.
4. Choose the commodity, variety, grade, and season.
5. Enter the temperature, rainfall, year, month, and day.
6. Submit the form.
7. View the predicted modal crop price.

## Model Information

The project uses a machine learning pipeline built with:

- Ordinal encoding for the grade feature
- One-hot encoding for categorical features like district, market, commodity, variety, and season
- Random Forest Regressor as the predictive model

This approach is suitable for estimating crop prices from structured agricultural market data.

## Data Used

The model is trained on historical agricultural market datasets stored in:

- `final_datset.csv`
- `final_datset_kharif.csv`

These datasets include features like district, market, commodity, grade, season, rainfall, temperature, and historical market price information.

## Deployment Notes

The application is developed for local deployment and is intended to run as a Flask application in a local environment. For production deployment, it may be adapted to a proper server environment such as:

- Gunicorn
- Nginx
- Cloud hosting (Heroku, Render, AWS, Azure, etc.)

## Future Enhancements

- Add more crops and regions
- Improve model accuracy using additional weather and economic indicators
- Add user authentication and admin dashboard
- Support multi-season training and more recent data ingestion
- Export predictions to CSV or PDF
- Add unit tests and CI/CD workflow

## Project Status

This project is a functional prototype for crop price prediction and demonstrates the use of machine learning for agricultural market forecasting.

## Credits

This project was developed as a mini-project focused on predictive analytics for agricultural commodity pricing.

## License

This project is provided for academic and educational use. Please review and comply with any institutional or repository-specific licensing terms before reuse or redistribution.
