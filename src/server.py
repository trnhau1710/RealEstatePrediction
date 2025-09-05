from fastapi import FastAPI
from pydantic import BaseModel
import os, joblib
import pandas as pd



# ======================
# 1. Load pipeline đã lưu
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # D:\RealEstateProject
pipeline_path = os.path.join(BASE_DIR, "models", "stack_pipeline.pkl")
stack_pipeline = joblib.load(pipeline_path)  

# ======================
# 2. Định nghĩa schema dữ liệu input
# ======================
class RealEstateInput(BaseModel):
    area: float
    floor_number: int
    bedroom_number: int
    is_dinning_room: int
    is_kitchen: int
    is_terrace: int
    is_car_park: int
    type: int                # nhận raw string
    street_width: float
    width: float
    city: str                # nhận raw string
    district_grouped: str    # nhận raw string

# ======================
# 3. Khởi tạo FastAPI app
# ======================
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Real Estate model deployment"}

@app.post("/predict")
def predict(data: RealEstateInput):
    try:
        # Convert input thành DataFrame để pipeline xử lý
        df = pd.DataFrame([data.model_dump()])

        # Gọi pipeline để dự đoán
        prediction = stack_pipeline.predict(df)[0]

        return {"prediction": float(prediction)}

    except Exception as e:
        return {"error": str(e)}
