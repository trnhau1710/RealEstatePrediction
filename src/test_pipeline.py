import os, joblib
import pandas as pd

def test_pipeline():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(BASE_DIR, "models")
    pipeline_path = os.path.join(models_dir, "stack_pipeline.pkl")

    # Load pipeline đã fit
    pipeline = joblib.load(pipeline_path)

    # Tạo 1 sample test (chú ý type là string, ví dụ "Nhà phố")
    test_data = pd.DataFrame([{
        "area": 83.0,
        "floor_number": 7.0,
        "bedroom_number": 7.0,
        "is_dinning_room": 1,
        "is_kitchen": 1,
        "is_terrace": 1,
        "is_car_park": 1,
        "type": 0,
        "street_width": 12.0,
        "width": 5.5,
        "city": "Hà Nội",
        "district_grouped": "Quận Cầu Giấy"
    }])

    # Thực hiện dự đoán
    y_pred = pipeline.predict(test_data)
    print("✅ Kết quả dự đoán cho sample:", y_pred)


if __name__ == "__main__":
    test_pipeline()