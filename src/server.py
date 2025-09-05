from fastapi import FastAPI
from pydantic import BaseModel
import os, joblib, gdown
import pandas as pd



# ======================
# 1. Load pipeline đã lưu
# ======================
MODEL_PATH = "stack_pipeline.pkl"

if not os.path.exists(MODEL_PATH):
    file_id = "1OOS4sujjakgfMho4Nxc4NLXVy_da5cjk"  # thay bằng ID thật
    gdown.download(f"https://drive.google.com/uc?id={file_id}", MODEL_PATH, quiet=False)

stack_pipeline = joblib.load(MODEL_PATH)


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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.server:app", host="0.0.0.0", port=8000)
