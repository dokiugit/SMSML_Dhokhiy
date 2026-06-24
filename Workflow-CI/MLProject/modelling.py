"""modelling.py - Training IndoBERT dengan MLflow (untuk MLProject).
Mendukung argumen CLI: --epochs, --lr, --batch_size
Jalankan via: mlflow run MLProject -P epochs=3 -P lr=0.00002 -P batch_size=16
"""
import argparse, pandas as pd, torch, mlflow, mlflow.pytorch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

MODEL_NAME = "indobenchmark/indobert-base-p1"
MAX_LEN    = 80
device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=3)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int,   default=16)
    args = parser.parse_args()

    tok      = BertTokenizer.from_pretrained(MODEL_NAME)
    train_df = pd.read_csv("dataset_preprocessing/train.csv")
    test_df  = pd.read_csv("dataset_preprocessing/test.csv")
    make     = lambda df, s: DataLoader(ReviewDataset(df["clean_review"].values,
                   df["label"].values, tok, MAX_LEN), batch_size=args.batch_size, shuffle=s)
    train_dl, test_dl = make(train_df, True), make(test_df, False)

    with mlflow.start_run():
        mlflow.log_params({"epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
                           "model": MODEL_NAME, "max_len": MAX_LEN})
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
        opt   = AdamW(model.parameters(), lr=args.lr)
        sched = get_linear_schedule_with_warmup(opt, 0, len(train_dl)*args.epochs)

        for ep in range(args.epochs):
            model.train(); total_loss = 0
            for b in train_dl:
                opt.zero_grad()
                out = model(input_ids=b["input_ids"].to(device),
                            attention_mask=b["attention_mask"].to(device),
                            labels=b["targets"].to(device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); total_loss += out.loss.item()
            avg_loss = total_loss / len(train_dl)
            mlflow.log_metric("train_loss", avg_loss, step=ep+1)
            print(f"Epoch {ep+1}/{args.epochs} | Loss: {avg_loss:.4f}")

        model.eval(); preds, labels = [], []
        with torch.no_grad():
            for b in test_dl:
                o = model(input_ids=b["input_ids"].to(device),
                          attention_mask=b["attention_mask"].to(device))
                preds.extend(torch.argmax(o.logits, 1).cpu().numpy())
                labels.extend(b["targets"].numpy())

        acc  = accuracy_score(labels, preds)
        prec = precision_score(labels, preds, average="weighted")
        rec  = recall_score(labels, preds, average="weighted")
        f1   = f1_score(labels, preds, average="weighted")

        mlflow.log_metrics({"accuracy": acc, "precision": prec, "recall": rec, "f1_score": f1})
        mlflow.pytorch.log_model(model, "model")

        print(f"\nTest Accuracy : {acc:.4f}")
        print(f"Precision     : {prec:.4f}")
        print(f"Recall        : {rec:.4f}")
        print(f"F1 Score      : {f1:.4f}")
        print(classification_report(labels, preds, target_names=["Negatif","Positif"]))

if __name__ == "__main__":
    main()
