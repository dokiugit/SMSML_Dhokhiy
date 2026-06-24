"""modelling_tuning.py - Hyperparameter tuning IndoBERT + manual logging MLflow (Skilled).
Artefak tambahan: confusion matrix & training curve.
Jalankan: python modelling_tuning.py
"""
import pandas as pd, numpy as np, torch, matplotlib.pyplot as plt, mlflow, mlflow.pytorch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, ConfusionMatrixDisplay)

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT   = "FinTech_Sentiment"
MODEL_NAME   = "indobenchmark/indobert-base-p1"
MAX_LEN      = 80
device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

PARAM_GRID = [
    {"epochs": 3, "lr": 2e-5, "batch_size": 16},
    {"epochs": 5, "lr": 3e-5, "batch_size": 16},
    {"epochs": 3, "lr": 2e-5, "batch_size": 32},
]

class ReviewDataset(Dataset):
    def __init__(self, r, t, tok, ml):
        self.r, self.t, self.tok, self.ml = r, t, tok, ml
    def __len__(self): return len(self.r)
    def __getitem__(self, i):
        e = self.tok.encode_plus(str(self.r[i]), add_special_tokens=True,
            max_length=self.ml, return_token_type_ids=False, padding="max_length",
            truncation=True, return_attention_mask=True, return_tensors="pt")
        return {"input_ids": e["input_ids"].flatten(),
                "attention_mask": e["attention_mask"].flatten(),
                "targets": torch.tensor(self.t[i], dtype=torch.long)}

def make_dl(df, tok, bs, shuf):
    return DataLoader(ReviewDataset(df["clean_review"].values, df["label"].values, tok, MAX_LEN),
                      batch_size=bs, shuffle=shuf)

def save_cm(y, p, name):
    ConfusionMatrixDisplay(confusion_matrix(y, p),
        display_labels=["Negatif","Positif"]).plot(cmap="Blues")
    plt.title(f"CM - {name}"); plt.tight_layout()
    plt.savefig("confusion_matrix.png"); plt.close()
    return "confusion_matrix.png"

def save_curve(hist, name):
    plt.figure(figsize=(7,4))
    plt.plot(hist["ep"], hist["loss"], "o-", label="Train Loss")
    plt.plot(hist["ep"], hist["acc"],  "s-", label="Val Acc")
    plt.title(f"Curve - {name}"); plt.xlabel("Epoch"); plt.legend()
    plt.tight_layout(); plt.savefig("training_curve.png"); plt.close()
    return "training_curve.png"

def run_exp(params, train_df, test_df, tok, run_name):
    train_dl = make_dl(train_df, tok, params["batch_size"], True)
    test_dl  = make_dl(test_df,  tok, params["batch_size"], False)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_params(params)
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
        opt   = AdamW(model.parameters(), lr=params["lr"])
        sched = get_linear_schedule_with_warmup(opt, 0, len(train_dl)*params["epochs"])
        hist  = {"ep": [], "loss": [], "acc": []}
        for ep in range(params["epochs"]):
            model.train(); tl = 0
            for b in train_dl:
                opt.zero_grad()
                out = model(input_ids=b["input_ids"].to(device),
                            attention_mask=b["attention_mask"].to(device),
                            labels=b["targets"].to(device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); tl += out.loss.item()
            model.eval(); ps, ts = [], []
            with torch.no_grad():
                for b in test_dl:
                    o = model(input_ids=b["input_ids"].to(device),
                              attention_mask=b["attention_mask"].to(device))
                    ps.extend(torch.argmax(o.logits,1).cpu().numpy())
                    ts.extend(b["targets"].numpy())
            loss = tl/len(train_dl); acc = accuracy_score(ts, ps)
            hist["ep"].append(ep+1); hist["loss"].append(loss); hist["acc"].append(acc)
            mlflow.log_metric("train_loss", loss, step=ep+1)
            mlflow.log_metric("val_accuracy", acc, step=ep+1)
        ps, ts = np.array(ps), np.array(ts)
        mlflow.log_metric("accuracy",  accuracy_score(ts, ps))
        mlflow.log_metric("precision", precision_score(ts, ps, average="weighted"))
        mlflow.log_metric("recall",    recall_score(ts, ps, average="weighted"))
        mlflow.log_metric("f1_score",  f1_score(ts, ps, average="weighted"))
        mlflow.pytorch.log_model(model, "model")
        mlflow.log_artifact(save_cm(ts, ps, run_name))
        mlflow.log_artifact(save_curve(hist, run_name))
        print(f"[{run_name}] acc={accuracy_score(ts,ps):.4f}")
        return accuracy_score(ts, ps)

def main():
    print("Memuat data & tokenizer...")
    train_df = pd.read_csv("dataset_preprocessing/train.csv")
    test_df  = pd.read_csv("dataset_preprocessing/test.csv")
    tok      = BertTokenizer.from_pretrained(MODEL_NAME)
    best_acc, best_p = 0, None
    for i, p in enumerate(PARAM_GRID):
        acc = run_exp(p, train_df, test_df, tok, f"IndoBERT_Tuning_Run{i+1}")
        if acc > best_acc: best_acc, best_p = acc, p
    print(f"\nBest Accuracy: {best_acc:.4f} | Best Params: {best_p}")
    print("Buka MLflow: http://127.0.0.1:5000")

if __name__ == "__main__": main()
