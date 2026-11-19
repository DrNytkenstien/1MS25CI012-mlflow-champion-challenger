import os
import json
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

EXPERIMENT_NAME = "hf-sentiment-champion-challenger"
REGISTERED_MODEL_NAME = "sentiment-model"
SUMMARY_PATH = "outputs/metrics_summary.json"

summary = None

# 1. Try reading local outputs/metrics_summary.json
if os.path.exists(SUMMARY_PATH) and os.path.getsize(SUMMARY_PATH) > 0:
    with open(SUMMARY_PATH, "r", encoding="utf-8") as f:
        content = f.read().strip()
        if content:
            summary = json.loads(content)

# 2. Fallback: fetch from MLflow server if local file is empty
if summary is None:
    print("Local metrics_summary.json is empty; fetching details from MLflow...")
    exp = client.get_experiment_by_name(EXPERIMENT_NAME)
    if exp is None:
        raise RuntimeError(f"Experiment '{EXPERIMENT_NAME}' not found. Run evaluate.py first.")

    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        order_by=["start_time DESC"],
    )

    compare_run = next((r for r in runs if r.info.run_name == "compare_sentiment"), None)
    run_a = next((r for r in runs if r.info.run_name.startswith("A_")), None)
    run_b = next((r for r in runs if r.info.run_name.startswith("B_")), None)

    if not (compare_run and run_a and run_b):
        raise RuntimeError("Required runs missing in MLflow. Run evaluate.py first.")

    try:
        os.makedirs("outputs", exist_ok=True)
        downloaded = mlflow.artifacts.download_artifacts(
            run_id=compare_run.info.run_id,
            artifact_path="metrics_summary.json",
            dst_path="outputs",
        )
        if os.path.exists(downloaded) and os.path.getsize(downloaded) > 0:
            with open(downloaded, "r", encoding="utf-8") as f:
                summary = json.load(f)
    except Exception:
        pass

    if summary is None:
        winner = compare_run.data.tags.get("winner", "A")
        reason = compare_run.data.tags.get("decision_reason", "Selected based on comparison rules.")

        info_a = dict(run_a.data.metrics)
        info_a["hf_model_id"] = run_a.data.params.get("hf_model_id", "distilbert-base-uncased-finetuned-sst-2-english")
        info_a["run_id"] = run_a.info.run_id
        info_a["model_uri"] = f"runs:/{run_a.info.run_id}/model"

        info_b = dict(run_b.data.metrics)
        info_b["hf_model_id"] = run_b.data.params.get("hf_model_id", "siebert/sentiment-roberta-large-english")
        info_b["run_id"] = run_b.info.run_id
        info_b["model_uri"] = f"runs:/{run_b.info.run_id}/model"

        summary = {
            "decision": {
                "primary_metric": compare_run.data.params.get("primary_metric", "macro_f1"),
                "margin": float(compare_run.data.params.get("margin", 0.02)),
                "winner": winner,
                "reason": reason,
            },
            "models": {
                "A": info_a,
                "B": info_b,
            },
        }

    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)
    print(f"Restored {SUMMARY_PATH} from MLflow.")

# 3. Register both models and assign @champion / @challenger aliases
winner = summary["decision"]["winner"]
models_info = summary["models"]

versions = {}
for role in ["A", "B"]:
    info = models_info[role]
    mv = mlflow.register_model(
        model_uri=info["model_uri"],
        name=REGISTERED_MODEL_NAME,
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=mv.version,
        key="hf_model_id",
        value=info["hf_model_id"],
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=mv.version,
        key="candidate_role",
        value=role,
    )
    versions[role] = mv.version
    print(f"Registered Model {role} ({info['hf_model_id']}) as {REGISTERED_MODEL_NAME} v{mv.version}")

champion_role = winner
challenger_role = "B" if winner == "A" else "A"

client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="champion",
    version=str(versions[champion_role]),
)
client.set_registered_model_alias(
    name=REGISTERED_MODEL_NAME,
    alias="challenger",
    version=str(versions[challenger_role]),
)

print(f"\nAlias '@champion'   -> v{versions[champion_role]} (Model {champion_role}: {models_info[champion_role]['hf_model_id']})")
print(f"Alias '@challenger' -> v{versions[challenger_role]} (Model {challenger_role}: {models_info[challenger_role]['hf_model_id']})")
