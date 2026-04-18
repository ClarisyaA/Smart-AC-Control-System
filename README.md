# Smart AC Control System

> Perbandingan **Manual FIS**, **GA-Tuned FIS**, dan **ANFIS** untuk kendali suhu ruangan cerdas  
> UTS Soft Computing — Genap 2025/2026 | Teknik Informatika, Universitas Padjadjaran  
> Dosen: Dr. Ir. Intan Nurma Yulita, M.T

---

## Deskripsi

Aplikasi Streamlit ini mengimplementasikan dan membandingkan tiga pendekatan *soft computing* untuk sistem kendali AC cerdas (*Smart AC Control*). Sistem mengambil input berupa kondisi ruangan secara real-time dan mengeluarkan rekomendasi kecepatan kompresor (0–100%) beserta estimasi konsumsi energi.

---

## Struktur Proyek

```
smart-ac-control/
├── app.py
└── artifacts/
    ├── dataset_smart_ac.csv       # Dataset 200 baris
    ├── metrics.json               # MSE / RMSE / MAE / R² ketiga model
    ├── energy_metrics.json        # Konsumsi energi ketiga model
    ├── predictions.json           # Prediksi seluruh dataset
    ├── ga_mf.json                 # Parameter MF hasil GA (trapezoid)
    ├── ga_history.json            # Riwayat konvergensi GA
    ├── anfis_mf.json              # Parameter Gaussian MF + consequents ANFIS
    ├── anfis_history.json         # Riwayat training loss ANFIS
    └── ablation/
        └── ablation_results.json  # Hasil 8 konfigurasi ablation GA
```

> **Penting:** Folder `artifacts/` harus dihasilkan terlebih dahulu dengan menjalankan seluruh sel notebook, lalu diletakkan di direktori yang sama dengan `app.py`.

---

## Instalasi

```bash
# Clone atau salin file proyek
git clone (https://github.com/ClarisyaA/Smart-AC-Control-System.git)
cd smart-ac-control-system

# Install dependensi
pip install -r requirements.txt
```

---

## Menjalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka di `http://localhost:8501`.

---

## Cara Penggunaan

Atur parameter ruangan melalui **sidebar kiri**:

| Parameter | Keterangan |
|---|---|
| Suhu Ruangan (°C) | Suhu terukur di dalam ruangan (15–38°C) |
| Suhu Target (°C) | Setpoint yang diinginkan pengguna (18–28°C) |
| Status Ruangan | Ada Penghuni / Kosong (jika kosong, output = 0%) |
| Suhu Luar (°C) | Referensi kondisi eksternal |
| Kelembaban (%RH) | Mempengaruhi beban kerja kompresor (30–90%) |
| Durasi Operasi (jam) | Untuk estimasi konsumsi energi |

Sidebar juga menampilkan **status ketersediaan** setiap file artifact.

---

## Tab dan Fitur

### 1. Ringkasan Utama
Inferensi real-time dari ketiga model sekaligus. Menampilkan kartu KPI kecepatan kompresor, grafik perbandingan, dan rekomendasi sistem berdasarkan nilai ensemble (rata-rata ketiga model). Tersedia juga ikhtisar dataset dengan distribusi kecepatan, energi, dan mode AC.

### 2. Fungsi Keanggotaan
Visualisasi MF untuk variabel input (*Selisih Suhu* dan *Kelembaban*). Mendukung tampilan individual per model maupun perbandingan overlay ketiganya. Tersedia tabel pergeseran parameter Manual → GA serta parameter Gaussian MF hasil training ANFIS.

### 3. Rule Base
Tampilan 15 rule Sugeno Zero-Order beserta kekuatan aktivasi (*firing strength*) setiap rule berdasarkan input saat ini. Menyertakan matriks rule (ΔT × Kelembaban), bar chart aktivasi, dan highlight rule dominan.

### 4. Analisis Performa
Kartu metrik MSE / RMSE / MAE / R² per model. Dilengkapi bar chart perbandingan metrik, scatter plot aktual vs prediksi, plot time-series 50 sampel pertama, kurva konvergensi GA, dan kurva training loss ANFIS.

### 5. Konsumsi Energi
Estimasi konsumsi energi real-time dan agregat dari dataset (200 sampel). Formula: `E = (Kecepatan/100) × 2.5 kW × durasi × faktor_kelembaban`. Tersedia scatter plot energi aktual vs prediksi dan grafik perbandingan total energi.

### 6. Ablation Study
Analisis sensitivitas 8 konfigurasi GA (variasi ukuran populasi dan jumlah generasi). Menampilkan tabel hasil, bar chart RMSE per konfigurasi, scatter pop size / gen vs RMSE, serta interpretasi fenomena konvergensi prematur dan *diminishing returns*.

---

## Model yang Dibandingkan

### Manual FIS
Sistem inferensi fuzzy Sugeno Zero-Order dengan **fungsi keanggotaan trapezoid** yang dirancang secara manual berdasarkan intuisi pakar. Menggunakan 15 rule dengan dua variabel input (ΔT dan Kelembaban).

### GA-Tuned FIS
Struktur rule identik dengan Manual FIS, namun **parameter trapezoid dioptimalkan** menggunakan Algoritma Genetika (default: Pop=60, Gen=120). GA memaksimalkan akurasi prediksi dengan meminimalkan RMSE + penalti validitas trapezoid.

### ANFIS
*Adaptive Neuro-Fuzzy Inference System* dengan **fungsi keanggotaan Gaussian** dan parameter consequent yang dioptimalkan menggunakan Adam optimizer (300 epoch). Seluruh parameter (center, sigma, consequent) dilatih end-to-end via backpropagation.

---

## Formula Energi

```
E (kWh) = (kecepatan / 100) × P_rated × t × faktor_kelembaban

P_rated        = 2.5 kW  (AC 1 PK standar)
faktor_kelembaban = 1 + 0.1 × (kelembaban / 50)
```

---

## Dependensi

| Package | Versi Minimum |
|---|---|
| Python | 3.8+ |
| streamlit | 1.20+ |
| numpy | 1.21+ |
| pandas | 1.3+ |
| matplotlib | 3.5+ |

---

## Catatan Teknis

- Jika ruangan **kosong** (`Okupansi = 0`), semua model mengembalikan output `0%` secara eksplisit — logika diimplementasikan sebelum inferensi fuzzy.
- ANFIS menggunakan **MinMax scaling** pada input (rentang dihitung dari dataset). Scaler direkonstruksi otomatis dari `dataset_smart_ac.csv` saat aplikasi dimuat.
- Output ANFIS dalam skala `[0, 1]` kemudian dikalikan 100 untuk konversi ke persen kecepatan.
- Seluruh artifact di-*cache* dengan `@st.cache_data` untuk performa optimal.
- Aplikasi berjalan sepenuhnya **offline** — tidak ada koneksi internet yang dibutuhkan setelah instalasi.
