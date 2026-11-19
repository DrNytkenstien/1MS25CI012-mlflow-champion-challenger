import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

REGISTERED_MODEL_NAME = "sentiment-model"
ALIAS = "champion"
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}@{ALIAS}"

# Fetch version and metadata via alias (no hardcoded version or run IDs)
mv = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, ALIAS)
run_data = client.get_run(mv.run_id).data
hf_model_id = mv.tags.get("hf_model_id") or run_data.params.get("hf_model_id", "unknown")

print(f"Loaded Alias   : @{ALIAS}")
print(f"Model Name     : {REGISTERED_MODEL_NAME}")
print(f"Model Version  : v{mv.version}")
print(f"HF Model ID    : {hf_model_id}")
print("-" * 75)

# Load registered champion model and run inference
champion_model = mlflow.pyfunc.load_model(MODEL_URI)
new_df = pd.read_csv("data/new_data.csv")
preds_df = champion_model.predict(new_df[["text"]])

preds_df["model_version"] = f"v{mv.version}"
preds_df["hf_model_id"] = hf_model_id

print(preds_df.to_string(index=False))

output_path = "outputs/new_data_predictions.csv"
preds_df.to_csv(output_path, index=False)
print(f"\nSaved predictions to {output_path}")