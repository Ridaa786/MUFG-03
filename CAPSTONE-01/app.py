from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import os
from io import StringIO

app = FastAPI(title="Manufacturing Output Prediction API")

# Load model only once at startup
model_path = os.path.join("models", "linear_regression_model.pkl")
pipeline = joblib.load(model_path)

# Import preprocess function from predict.py OR duplicate logic (we will reuse)
from src.predict import preprocess_new_data

@app.post("/predict_json")
async def predict_json(data: dict):
    # Convert JSON to DataFrame
    df = pd.DataFrame([data])

    # Preprocess data
    df_processed = preprocess_new_data(df)

    # Reorder columns
    df_processed = df_processed[pipeline.feature_names_in_]

    # Predict
    prediction = pipeline.predict(df_processed)[0]

    return {"Predicted_Parts_Per_Hour": prediction}


@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(StringIO(contents.decode("utf-8")))

    # Preprocess data
    df_processed = preprocess_new_data(df)

    # Reorder columns
    df_processed = df_processed[pipeline.feature_names_in_]

    # Predict
    predictions = pipeline.predict(df_processed)

    # Append predictions back to original dataframe
    df["Predicted_Parts_Per_Hour"] = predictions

    # Return sample rows (for checking)
    return {
        "message": "✅ Predictions generated successfully",
        "sample_results": df.head(5).to_dict(orient="records")
    }


@app.get("/")
def root():
    return {"message": "✅ Manufacturing Output Prediction API is running"}
