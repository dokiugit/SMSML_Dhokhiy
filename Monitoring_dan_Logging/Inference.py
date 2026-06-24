"""7.inference.py - Inferensi sentimen ulasan FinTech via IndoBERT MLflow.
Jalankan: python Inference.py
"""
import re, torch, mlflow, mlflow.pytorch
from transformers import BertTokenizer

TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT   = "FinTech_Sentiment"
MODEL_NAME   = "indobenchmark/indobert-base-p1"
MAX_LEN      = 80
CLASS_NAMES  = ["Negatif", "Positif"]
mlflow.set_tracking_uri(TRACKING_URI)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_latest_model():
    client = mlflow.tracking.MlflowClient()
    exp    = client.get_experiment_by_name(EXPERIMENT)
    runs   = client.search_runs([exp.experiment_id], order_by=["start_time DESC"], max_results=1)
    if not runs: raise ValueError("Tidak ada model. Jalankan modelling.py dulu!")
    model = mlflow.pytorch.load_model(f"runs:/{runs[0].info.run_id}/model", map_location=device)
    return model.eval()

def clean(text):
    t = str(text).lower()
    t = re.sub(r"https?://\S+", "", t)
    t = re.sub(r"[^\w\s]", "", t).strip()
    return t

def predict(text, model, tokenizer):
    enc = tokenizer.encode_plus(clean(text), add_special_tokens=True, max_length=MAX_LEN,
          return_token_type_ids=False, padding="max_length", truncation=True,
          return_attention_mask=True, return_tensors="pt")
    with torch.no_grad():
        out   = model(input_ids=enc["input_ids"].to(device),
                      attention_mask=enc["attention_mask"].to(device))
        proba = torch.softmax(out.logits, 1).cpu().numpy()[0]
        pred  = int(proba.argmax())
    return {"prediksi": CLASS_NAMES[pred], "confidence": round(float(proba[pred]), 4),
            "prob": {c: round(float(p), 4) for c, p in zip(CLASS_NAMES, proba)}}

if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model     = load_latest_model()
    contoh    = [
        "Aplikasi ini sangat mudah digunakan dan transfernya cepat!",
        "Sering error dan uang tidak masuk, sangat mengecewakan.",
    ]
    for t in contoh:
        h = predict(t, model, tokenizer)
        print(f"Teks      : {t}")
        print(f"Sentimen  : {h['prediksi']}  (conf: {h['confidence']:.2%})")
        print(f"Prob      : {h['prob']}\n")
