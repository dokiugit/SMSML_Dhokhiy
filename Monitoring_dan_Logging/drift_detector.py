"""
Modul Deteksi Data Drift menggunakan Population Stability Index (PSI)

PSI mengukur seberapa besar pergeseran distribusi data produksi dibandingkan
distribusi data training (referensi). Ini adalah teknik industri yang digunakan
di perbankan, asuransi, dan sistem ML skala besar.

Formula PSI:
    PSI = Σ (P_actual - P_expected) × ln(P_actual / P_expected)

Interpretasi:
    PSI < 0.10  → Tidak ada drift signifikan (aman)
    0.10–0.20   → Drift moderat (perlu dimonitor)
    PSI > 0.20  → Drift signifikan (pertimbangkan retraining!)
"""
import numpy as np
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    n_bins: int = 10,
    eps: float = 1e-6,
) -> float:
    """
    Hitung Population Stability Index antara distribusi referensi dan saat ini.

    Args:
        reference: Data distribusi referensi (dari training set).
        current:   Data distribusi produksi (dari live inference window).
        n_bins:    Jumlah bucket (bin) untuk membagi distribusi.
        eps:       Nilai kecil untuk mencegah pembagian dengan nol.

    Returns:
        Nilai PSI (float). Semakin tinggi = semakin besar drift.
    """
    if len(reference) == 0 or len(current) == 0:
        logger.warning("Array kosong diberikan ke compute_psi, mengembalikan 0.0")
        return 0.0

    # Tentukan batas bin berdasarkan distribusi referensi
    breakpoints = np.linspace(
        min(reference.min(), current.min()),
        max(reference.max(), current.max()),
        n_bins + 1,
    )

    # Hitung proporsi per bin
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)

    ref_pct = ref_counts / (len(reference) + eps)
    cur_pct = cur_counts / (len(current) + eps)

    # Hindari log(0): tambahkan eps ke proporsi nol
    ref_pct = np.where(ref_pct == 0, eps, ref_pct)
    cur_pct = np.where(cur_pct == 0, eps, cur_pct)

    # PSI per bin, kemudian sum
    psi_per_bin = (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    psi = float(np.sum(psi_per_bin))

    return psi


def interpret_psi(psi: float) -> Tuple[str, str]:
    """
    Terjemahkan nilai PSI ke status dan saran tindakan.

    Returns:
        (status, action) — keduanya berupa string.
    """
    if psi < 0.10:
        return "stable", "Distribusi data stabil. Tidak diperlukan tindakan."
    elif psi < 0.20:
        return "warning", "Drift moderat terdeteksi. Pantau lebih sering dan evaluasi model."
    else:
        return "critical", "Drift signifikan! Pertimbangkan retraining model segera."


class FeatureDriftDetector:
    """
    Detektor drift per fitur menggunakan PSI.
    Menyimpan distribusi referensi dari training data dan menghitung
    PSI terhadap window data produksi yang masuk.
    """

    def __init__(self, feature_names: List[str], window_size: int = 100):
        """
        Args:
            feature_names: Daftar nama fitur yang akan dipantau.
            window_size:   Jumlah sampel per window untuk menghitung PSI.
        """
        self.feature_names = feature_names
        self.window_size = window_size
        self.reference_data: dict[str, np.ndarray] = {}
        self.current_window: dict[str, List[float]] = {f: [] for f in feature_names}
        self.psi_scores: dict[str, float] = {f: 0.0 for f in feature_names}
        self.window_count = 0

    def set_reference(self, reference_data: dict[str, np.ndarray]) -> None:
        """
        Simpan distribusi referensi dari data training.

        Args:
            reference_data: Dict {feature_name: array_of_values}.
        """
        self.reference_data = {
            name: np.array(values) for name, values in reference_data.items()
        }
        logger.info(
            f"Distribusi referensi ditetapkan untuk {len(reference_data)} fitur: "
            f"{list(reference_data.keys())}"
        )

    def add_sample(self, sample: dict[str, float]) -> None:
        """
        Tambahkan satu sampel produksi ke window saat ini.

        Args:
            sample: Dict {feature_name: value}.
        """
        for feature, value in sample.items():
            if feature in self.current_window:
                self.current_window[feature].append(value)

    def compute_window_psi(self) -> dict[str, float]:
        """
        Hitung PSI untuk semua fitur menggunakan data window saat ini.
        Otomatis reset window setelah perhitungan.

        Returns:
            Dict {feature_name: psi_score}.
        """
        if not self.reference_data:
            logger.warning("Distribusi referensi belum ditetapkan. Panggil set_reference() terlebih dahulu.")
            return self.psi_scores

        window_too_small = all(
            len(self.current_window[f]) < self.window_size // 2
            for f in self.feature_names
        )
        if window_too_small:
            return self.psi_scores

        results = {}
        for feature in self.feature_names:
            current = np.array(self.current_window[feature])
            reference = self.reference_data.get(feature)

            if reference is None or len(current) == 0:
                results[feature] = 0.0
                continue

            psi = compute_psi(reference, current)
            status, action = interpret_psi(psi)
            results[feature] = psi

            if status == "critical":
                logger.warning(
                    f"[DRIFT ALERT] Fitur '{feature}': PSI={psi:.4f} — {action}"
                )
            elif status == "warning":
                logger.info(
                    f"[DRIFT WARNING] Fitur '{feature}': PSI={psi:.4f} — {action}"
                )
            else:
                logger.debug(f"Fitur '{feature}': PSI={psi:.4f} — {action}")

        self.psi_scores = results
        self.window_count += 1

        # Reset window setelah dihitung
        self.current_window = {f: [] for f in self.feature_names}

        return results

    def get_overall_psi(self) -> float:
        """Rata-rata PSI semua fitur sebagai skor drift keseluruhan."""
        if not self.psi_scores:
            return 0.0
        return float(np.mean(list(self.psi_scores.values())))

    def summary(self) -> str:
        """Ringkasan teks status drift semua fitur."""
        if not self.psi_scores:
            return "Belum ada data PSI."
        lines = [f"=== Drift Summary (Window #{self.window_count}) ==="]
        for feature, psi in self.psi_scores.items():
            status, _ = interpret_psi(psi)
            icon = {"stable": "OK", "warning": "WARN", "critical": "CRIT"}[status]
            lines.append(f"  [{icon}] {feature:<20} PSI = {psi:.4f}")
        lines.append(f"  Overall PSI: {self.get_overall_psi():.4f}")
        return "\n".join(lines)
