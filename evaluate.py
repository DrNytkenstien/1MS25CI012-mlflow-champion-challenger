import os
import json
import time
import inspect
import torch
import transformers
import mlflow
import mlflow.pyfunc
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# 1. Custom MLflow PyFunc Wrapper (Required for Step 5.3 & 5.6)
class SentimentPyfuncModel(mlflow.pyfunc.PythonModel):
    def __init__(self, hf_model_id: str, max_length: int = 512):
        self.hf_model_id = hf_model_id
        self.max_length = max_length

    def load_context(self, context):
        from transformers import pipeline
        self.pipe = pipeline("sentiment-analysis", model=self.hf_model_id, device=-1)

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            texts = model_input["text"].tolist()
        else:
            texts = list(model_input)
        outputs = self.pipe(texts, truncation=True, max_length=self.max_length)
        return pd.DataFrame({
            "input": texts,
            "prediction": [o["label"] for o in outputs],
            "confidence": [round(float(o["score"]), 4) for o in outputs],
        })


# 2. MLflow Setup
os.makedirs("outputs", exist_ok=True)
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("hf-sentiment-champion-challenger")

SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 8
STUDENT_USN = "1MS25CI012"
DATASET_NAME = "custom_sentiment_eval_10"

# 3. Load Evaluation Data
print("Loading data...")
df = pd.read_csv("data/eval_inputs.csv")
texts = df["text"].tolist()
true_labels = df["label"].tolist()

models = {
    "A": {
        "name": "distilbert-sst2",
        "id": "distilbert-base-uncased-finetuned-sst-2-english",
        "role": "champion",
    },
    "B": {
        "name": "roberta-large",
        "id": "siebert/sentiment-roberta-large-english",
        "role": "challenger candidate",
    },
}

results = {}
all_preds = {}
lib_versions = f"transformers={transformers.__version__};torch={torch.__version__};mlflow={mlflow.__version__}"

# 4. Run and Log Both Models
for role, m in models.items():
    run_name = f"{role}_sentiment_{m['name']}"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n--- Running {run_name} ---")
        torch.manual_seed(SEED)

        pipe = pipeline("sentiment-analysis", model=m["id"], device=-1)

        start_time = time.perf_counter()
        preds = pipe(texts, truncation=True, max_length=MAX_LENGTH, batch_size=BATCH_SIZE)
        total_inference_s = time.perf_counter() - start_time
        avg_latency_ms = (total_inference_s / len(texts)) * 1000

        pred_labels = [p["label"] for p in preds]
        scores = [round(float(p["score"]), 4) for p in preds]
        all_preds[role] = pred_labels

        acc = accuracy_score(true_labels, pred_labels)
        prec, rec, macro_f1, _ = precision_recall_fscore_support(
            true_labels, pred_labels, average="macro", zero_division=0
        )

        # Log Parameters
        mlflow.log_params({
            "hf_model_id": m["id"],
            "task": "sentiment-classification",
            "role": m["role"],
            "max_length": MAX_LENGTH,
            "batch_size": BATCH_SIZE,
            "eval_samples": len(texts),
            "seed": SEED,
            "device": "cpu",
        })

        # Log Tags
        mlflow.set_tags({
            "student_usn": STUDENT_USN,
            "library_versions": lib_versions,
            "dataset_name": DATASET_NAME,
        })

        # Log Metrics
        mlflow.log_metrics({
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "avg_latency_ms": round(float(avg_latency_ms), 2),
            "total_inference_s": round(float(total_inference_s), 4),
        })

        # Save and Log Predictions Artifact
        out_df = pd.DataFrame({
            "input": texts,
            "reference": true_labels,
            "prediction": pred_labels,
            "confidence": scores,
            "correct": [t == p for t, p in zip(true_labels, pred_labels)],
        })
        pred_file = f"outputs/predictions_{role}.csv"
        out_df.to_csv(pred_file, index=False)
        mlflow.log_artifact(pred_file)

        # Log Wrapped Model into MLflow
        pyfunc_model = SentimentPyfuncModel(hf_model_id=m["id"], max_length=MAX_LENGTH)
        log_kwargs = {
            "python_model": pyfunc_model,
            "input_example": pd.DataFrame({"text": texts[:2]}),
        }
        if "name" in inspect.signature(mlflow.pyfunc.log_model).parameters:
            log_kwargs["name"] = "model"
        else:
            log_kwargs["artifact_path"] = "model"

        model_info = mlflow.pyfunc.log_model(**log_kwargs)

        results[role] = {
            "hf_model_id": m["id"],
            "run_id": run.info.run_id,
            "model_uri": model_info.model_uri,
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "macro_f1": round(float(macro_f1), 4),
            "avg_latency_ms": round(float(avg_latency_ms), 2),
            "total_inference_s": round(float(total_inference_s), 4),
        }
        print(f"Macro-F1: {macro_f1:.4f} | Accuracy: {acc:.4f} | Avg Latency: {avg_latency_ms:.2f}ms")

# 5. Generate Evaluation Plots & Qualitative Disagreements
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
labels_order = ["NEGATIVE", "POSITIVE"]
for idx, (role, title) in enumerate([("A", "Model A: DistilBERT"), ("B", "Model B: RoBERTa-Large")]):
    cm = confusion_matrix(true_labels, all_preds[role], labels=labels_order)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_order)
    disp.plot(ax=axes[idx], colorbar=False)
    axes[idx].set_title(title)
plt.tight_layout()
cm_plot_path = "outputs/confusion_matrices.png"
plt.savefig(cm_plot_path, dpi=150)
plt.close()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
metric_names = ["Accuracy", "Macro-F1"]
a_scores = [results["A"]["accuracy"], results["A"]["macro_f1"]]
b_scores = [results["B"]["accuracy"], results["B"]["macro_f1"]]
x = range(len(metric_names))
ax1.bar([i - 0.15 for i in x], a_scores, width=0.3, label="Model A (Champion)")
ax1.bar([i + 0.15 for i in x], b_scores, width=0.3, label="Model B (Challenger)")
ax1.set_xticks(list(x))
ax1.set_xticklabels(metric_names)
ax1.set_ylim(0, 1.1)
ax1.set_title("Quality Metrics (Higher is Better)")
ax1.legend()

ax2.bar(["Model A", "Model B"], [results["A"]["avg_latency_ms"], results["B"]["avg_latency_ms"]], color=["#1f77b4", "#ff7f0e"])
ax2.set_ylabel("ms / sample")
ax2.set_title("Avg Latency (Lower is Better)")
plt.tight_layout()
comp_plot_path = "outputs/model_comparison.png"
plt.savefig(comp_plot_path, dpi=150)
plt.close()

disagree_df = pd.DataFrame({
    "input": texts,
    "reference": true_labels,
    "model_A_pred": all_preds["A"],
    "model_B_pred": all_preds["B"],
    "models_disagree": [a != b for a, b in zip(all_preds["A"], all_preds["B"])],
})
disagree_path = "outputs/disagreements.csv"
disagree_df.to_csv(disagree_path, index=False)

# 6. Champion-Challenger Decision Summary Run
with mlflow.start_run(run_name="compare_sentiment"):
    print("\n--- Running Champion-Challenger Comparison ---")
    PRIMARY_METRIC = "macro_f1"
    MARGIN = 0.02  # Challenger must improve macro-F1 by at least 2 percentage points

    delta_primary = round(results["B"][PRIMARY_METRIC] - results["A"][PRIMARY_METRIC], 4)
    delta_accuracy = round(results["B"]["accuracy"] - results["A"]["accuracy"], 4)

    # Decision Rule:
    # 1. Primary metric: macro_f1 (higher is better), minimum improvement margin = 0.02.
    # 2. Guardrail: reject Challenger if latency is > 5x Champion's latency.
    # 3. Tie-breaker: if within margin, keep the faster model (lower avg_latency_ms).
    latency_ratio = results["B"]["avg_latency_ms"] / max(results["A"]["avg_latency_ms"], 0.01)

    if delta_primary >= MARGIN and latency_ratio <= 5.0:
        winner = "B"
        reason = f"Challenger improved macro_f1 by {delta_primary:+.4f} (>= {MARGIN}) within latency guardrail ({latency_ratio:.2f}x)."
    elif abs(delta_primary) < MARGIN and results["B"]["avg_latency_ms"] < results["A"]["avg_latency_ms"]:
        winner = "B"
        reason = "Within margin, and Challenger was faster."
    else:
        winner = "A"
        reason = f"Champion retained (delta macro_f1 = {delta_primary:+.4f}, margin required = {MARGIN}, latency ratio = {latency_ratio:.2f}x)."

    mlflow.log_params({
        "primary_metric": PRIMARY_METRIC,
        "margin": MARGIN,
        "direction": "higher_is_better",
        "latency_guardrail_max_ratio": 5.0,
    })
    mlflow.set_tags({
        "winner": winner,
        "decision_reason": reason,
        "student_usn": STUDENT_USN,
    })
    mlflow.log_metrics({
        "delta_primary_metric": delta_primary,
        "delta_accuracy": delta_accuracy,
        "latency_ratio_B_over_A": round(latency_ratio, 2),
    })

    summary_payload = {
        "decision": {
            "primary_metric": PRIMARY_METRIC,
            "margin": MARGIN,
            "winner": winner,
            "reason": reason,
        },
        "models": results,
    }