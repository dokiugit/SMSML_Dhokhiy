# Workflow CI - Kriteria 3

## Cara Upload ke GitHub

1. Buat repository baru di GitHub bernama `Workflow-CI_Nama-siswa` (public)
2. Copy semua file dari folder ini ke repo tersebut:
   - `.github/workflows/ci.yml` — workflow utama
   - Semua file dari `submission-files/` (automate_siswa.py, Membangun_model/, Monitoring_dan_Logging/)
3. Push ke branch `main` atau `master`
4. Buka tab **Actions** di GitHub untuk melihat CI berjalan otomatis
5. Salin link repo ke dalam `Workflow-CI.txt`

## Struktur Repo GitHub

```
Workflow-CI_Nama-siswa/
├── .github/
│   └── workflows/
│       └── ci.yml          ← Workflow CI utama
├── automate_siswa.py
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   └── requirements.txt
└── Monitoring_dan_Logging/
    ├── prometheus.yml
    ├── prometheus_exporter.py
    └── Inference.py
```

## Jobs dalam CI Pipeline

| Job | Fungsi |
|-----|--------|
| `data-preprocessing` | Menjalankan `automate_siswa.py` dengan dataset sample, verifikasi output |
| `model-training` | Install deps, train model, catat ke MLflow, verifikasi run |
| `code-quality` | Flake8 + black formatting check + syntax validation |
| `summary` | Ringkasan status semua job |
