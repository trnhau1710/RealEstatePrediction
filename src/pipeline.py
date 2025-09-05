import os, joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

def create_pipeline():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(BASE_DIR, "models")

    # Load stacking model đã train sẵn
    stack_model = joblib.load(os.path.join(models_dir, "stack_model.pkl"))

    # Các cột
    num_cols = ["area", "floor_number", "bedroom_number", "street_width", "width"]

    # ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("city_enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["city"]),
            ("district_enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ["district_grouped"]),
            ("num_scaler", StandardScaler(), num_cols)
        ],
        remainder="passthrough"  # không giữ cột nào thừa
    )

    # Ghép vào pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("stacking", stack_model)
    ])
    return pipeline


if __name__ == "__main__":
    pipeline = create_pipeline()

    # Load dữ liệu train gốc để fit preprocessor
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(BASE_DIR, "data", "data_visualized.csv")
    df = pd.read_csv(data_path)

    # Loại bỏ cột không dùng
    drop_cols = ["price"]
    X = df.drop(columns=drop_cols, errors="ignore")
    y = df["price"]

    # Fit pipeline (preprocessor sẽ học encoder + scaler)
    pipeline.fit(X, y)

    # Lưu pipeline đã fit
    models_dir = os.path.join(BASE_DIR, "models")
    pipeline_path = os.path.join(models_dir, "stack_pipeline.pkl")
    joblib.dump(pipeline, pipeline_path)
    print(f"✅ Pipeline đã fit và lưu tại: {pipeline_path}")