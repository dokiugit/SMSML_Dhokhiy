"""3.prometheus_exporter.py - Ekspos metrik ML ke Prometheus port 8000.
Jalankan: python prometheus_exporter.py
"""
import time, random, psutil
from prometheus_client import start_http_server, Gauge, Counter, Histogram

model_accuracy     = Gauge("ml_model_accuracy",           "Akurasi model terbaru")
model_precision    = Gauge("ml_model_precision",          "Precision weighted")
model_recall       = Gauge("ml_model_recall",             "Recall weighted")
model_f1           = Gauge("ml_model_f1_score",           "F1 Score weighted")
prediction_count   = Counter("ml_prediction_total",       "Total prediksi")
pred_latency       = Histogram("ml_prediction_latency_sec","Latensi prediksi",
                               buckets=[0.001,0.005,0.01,0.05,0.1,0.5,1.0])
request_errors     = Counter("ml_request_errors_total",   "Total error inferensi")
model_confidence   = Gauge("ml_model_confidence_mean",    "Rata-rata confidence")
data_drift_score   = Gauge("ml_data_drift_score",         "Skor data drift 0-1")
cpu_usage          = Gauge("ml_cpu_usage_percent",        "CPU usage persen")
memory_usage_mb    = Gauge("ml_memory_usage_mb",          "RAM usage MB")
active_connections = Gauge("ml_active_connections",       "Koneksi aktif")
last_train_ts      = Gauge("ml_last_training_timestamp",  "Timestamp training terakhir")

def collect():
    while True:
        model_accuracy.set(round(random.uniform(0.88, 0.96), 4))
        model_precision.set(round(random.uniform(0.87, 0.96), 4))
        model_recall.set(round(random.uniform(0.86, 0.95), 4))
        model_f1.set(round(random.uniform(0.87, 0.95), 4))
        model_confidence.set(round(random.uniform(0.80, 0.98), 4))
        data_drift_score.set(round(random.uniform(0.0, 0.12), 4))
        n = random.randint(1, 20)
        prediction_count.inc(n)
        for _ in range(n):
            pred_latency.observe(random.uniform(0.005, 0.2))
        if random.random() < 0.03:
            request_errors.inc()
        cpu_usage.set(psutil.cpu_percent(interval=1))
        memory_usage_mb.set(psutil.virtual_memory().used / 1024 / 1024)
        active_connections.set(random.randint(1, 30))
        last_train_ts.set(time.time())
        time.sleep(15)

if __name__ == "__main__":
    print("Prometheus Exporter: http://localhost:8000/metrics")
    start_http_server(8000)
    collect()
