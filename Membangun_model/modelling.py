"""modelling.py - Training IndoBERT dengan MLflow autolog (Basic).
Jalankan: python modelling.py
"""
import pandas as pd, torch, mlflow, mlflow.pytorch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT   = "FinTech_Sentiment"
MODEL_NAME   = "indobenchmark/indobert-base-p1"
MAX_LEN, BATCH, EPOCHS, LR = 80, 16, 3, 2e-5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT)

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

def train():
    tok      = BertTokenizer.from_pretrained(MODEL_NAME)
    train_df = pd.read_csv("dataset_preprocessing/train.csv")
    test_df  = pd.read_csv("dataset_preprocessing/test.csv")
    make     = lambda df, s: DataLoader(ReviewDataset(df["clean_review"].values,
                   df["label"].values, tok, MAX_LEN), batch_size=BATCH, shuffle=s)
    train_dl, test_dl = make(train_df, True), make(test_df, False)
    mlflow.pytorch.autolog()
    with mlflow.start_run(run_name="IndoBERT_Basic"):
        model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2).to(device)
        opt   = AdamW(model.parameters(), lr=LR)
        sched = get_linear_schedule_with_warmup(opt, 0, len(train_dl)*EPOCHS)
        for ep in range(EPOCHS):
            model.train()
            for b in train_dl:
                opt.zero_grad()
                out = model(input_ids=b["input_ids"].to(device),
                            attention_mask=b["attention_mask"].to(device),
                            labels=b["targets"].to(device))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
            print(f"Epoch {ep+1}/{EPOCHS} selesai")
        model.eval(); preds = []
        with torch.no_grad():
            for b in test_dl:
                o = model(input_ids=b["input_ids"].to(device),
                          attention_mask=b["attention_mask"].to(device))
                preds.extend(torch.argmax(o.logits, 1).cpu().numpy())
        y_true = test_df["label"].values
        print(f"\nTest Accuracy: {accuracy_score(y_true, preds):.4f}")
        print(classification_report(y_true, preds, target_names=["Negatif","Positif"]))
        print("Buka MLflow: http://127.0.0.1:5000")

if __name__ == "__main__":
    train()
