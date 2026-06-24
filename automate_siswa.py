"""automate_siswa.py - Pipeline preprocessing otomatis FinTech Ulasan.
Alias dari automate_NamaSiswa.py — gunakan salah satu sesuai konvensi penamaan.
Jalankan: python automate_siswa.py
"""
import pandas as pd, numpy as np, re, os
from sklearn.model_selection import train_test_split

DEFAULT_URL = (
    "https://docs.google.com/spreadsheets/d/e/"
    "2PACX-1vR3MkBjEU0LxWGqKacBiDbr62KlHfensMl59SSFSST4GIX9zD83C4jebFPDOehuy5hraqbI6KuIlSIT"
    "/pub?output=csv"
)

def load_data(url=None, local_path=None):
    if local_path and os.path.exists(local_path):
        df = pd.read_csv(local_path)
    else:
        df = pd.read_csv(url or DEFAULT_URL)
    print(f"[1/5] Data dimuat. Shape: {df.shape}")
    return df

def check_and_clean(df):
    before = len(df)
    df = df.dropna(subset=["review", "rating"]).drop_duplicates()
    print(f"[2/5] Cleaning: {before - len(df)} baris dihapus. Sisa: {len(df)}")
    return df

def label_and_balance(df):
    df = df[df["rating"] != 3].copy()
    df["label"] = df["rating"].apply(lambda x: 1 if x > 3 else 0)
    n = df["label"].value_counts().min()
    df = (pd.concat([df[df["label"]==1].sample(n, random_state=42),
                     df[df["label"]==0].sample(n, random_state=42)])
          .sample(frac=1, random_state=42).reset_index(drop=True))
    print(f"[3/5] Balance: {df['label'].value_counts().to_dict()}")
    return df

def clean_text(text):
    t = str(text).lower()
    t = re.sub(r"https?://\S+|www\.\S+", "", t)
    t = re.sub(r"[-+]?[0-9]+", "", t)
    t = re.sub(r"[^\w\s]", "", t).strip()
    return t

def preprocess_text(df):
    df = df.copy()
    df["clean_review"] = df["review"].apply(clean_text)
    before = len(df)
    df = df[df["clean_review"].str.strip() != ""]
    print(f"[4/5] Teks bersih. {before - len(df)} baris kosong dihapus.")
    return df

def save_data(df, output_dir="dataset_preprocessing"):
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(f"{output_dir}/dataset_preprocessed.csv", index=False)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
    train.to_csv(f"{output_dir}/train.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)
    print(f"[5/5] Disimpan ke '{output_dir}' | total={len(df)} | train={len(train)} | test={len(test)}")

def run_preprocessing(url=None, local_path=None, output_dir="dataset_preprocessing"):
    print("=" * 50)
    print("  AUTOMATE PREPROCESSING - FinTech Dataset")
    print("=" * 50)
    df = load_data(url=url, local_path=local_path)
    df = check_and_clean(df)
    df = label_and_balance(df)
    df = preprocess_text(df)
    save_data(df, output_dir=output_dir)
    print("=" * 50)
    print("  Selesai! Data siap dilatih.")
    print("=" * 50)
    return df

if __name__ == "__main__":
    run_preprocessing()
