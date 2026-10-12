import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

# Load the dataset
df = pd.read_csv("D:\\False_dataset\\Final_datset\\final_datset_kharif.csv")

# Drop Min/Max Prices from the main dataset for Modal Price model
df.drop(['Sl no.','Min Price (Rs./Quintal)', 'Max Price (Rs./Quintal)', 'Price Date','Humidity (%)'], axis=1, inplace=True)

# Define preprocessing
grade_order = ['Small', 'Medium', 'FAQ', 'Large']
oe = OrdinalEncoder(categories=[grade_order])
ohe = OneHotEncoder(drop='first', dtype=int)

preprocessor = ColumnTransformer(
    transformers=[
        ('ordinal', oe, ['Grade']),  # Ordinal Encoding for 'Grade'
        ('onehot', ohe, ['District Name', 'Market Name', 'Commodity', 'Variety','Season'])  # One-Hot Encoding
    ],
    remainder='passthrough'  # Pass other features unchanged
)

# Prepare data for Modal Price model
X = df.drop('Modal Price (Rs./Quintal)', axis=1)
y = df['Modal Price (Rs./Quintal)']

# Create pipeline for Modal Price prediction
modal_price_pipeline = Pipeline(
    steps=[
        ('preprocessor', preprocessor),
        ('model', RandomForestRegressor(n_estimators=150, random_state=42))
    ]
)

# Train the Modal Price pipeline
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
modal_price_pipeline.fit(X_train, y_train)

# Save the Modal Price pipeline
joblib.dump(modal_price_pipeline, 'manju_final.pkl')

# Prediction function
def predict_modal_price(input_data):
    # Load the pipeline
    modal_pipeline = joblib.load('manju_final.pkl')
    
    # Preprocess input data
    preprocessor = modal_pipeline.named_steps['preprocessor']
    
    # Check and fix 'Grade' if necessary
    grade_order = ['Small', 'Medium', 'FAQ', 'Large']
    input_data['Grade'] = input_data['Grade'].apply(lambda x: x if x in grade_order else 'Medium')  # Default to 'Medium' if unknown
    
    input_data_preprocessed = preprocessor.transform(input_data)
    
    # Predict Modal Price
    return modal_pipeline.named_steps['model'].predict(input_data_preprocessed)

# Example Input Data
input_data = pd.DataFrame([[
    'Chittor',          # District Name
    'Bangarupalem',      # Market Name
    'Mango',             # Commodity
    'Neelam',            # Variety
    'Medium',            # Grade
    'Kharif',            # Season
    22.0,                # Temperature (°C)
    153.2,               # Rainfall (mm)
    2013,                # Year
    6,                   # Month
    29                  # Day
]], columns=[
    'District Name', 
    'Market Name', 
    'Commodity', 
    'Variety', 
    'Grade', 
    'Season',
    'Temperature (°C)', 
    'Rainfall (mm)', 
    'Year', 
    'Month', 
    'Day'
])

# Predict
predicted_price = predict_modal_price(input_data)
print(f"Predicted Modal Price: {predicted_price[0]}")
