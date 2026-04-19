# Insurance Charges Prediction Using FIS, GA Tuning, and NeuroFuzzy ANN

## 1. Deskripsi Aplikasi

Aplikasi ini merupakan sistem prediksi biaya asuransi kesehatan berbasis antarmuka Streamlit.
Pengguna dapat memasukkan nilai usia, BMI, dan status perokok, lalu sistem menampilkan estimasi biaya asuransi.

Pengembangan aplikasi dilakukan untuk membandingkan kinerja tiga pendekatan:

1. FIS Manual (berbasis intuisi pakar)
2. FIS yang dituning menggunakan Genetic Algorithm (GA)
3. NeuroFuzzy ANN (ANFIS-inspired) berbasis PyTorch

## 2. Metode yang Digunakan

### 2.1 Tahap 1 - Manual FIS (Sugeno Zero-Order)

Sistem fuzzy awal dibangun secara manual dengan membership function Gaussian dan rule base berdasarkan intuisi pakar.

### 2.2 Tahap 2 - Evolutionary Tuning dengan GA

Parameter membership function dan konsekuen rule pada FIS dioptimasi menggunakan real-coded Genetic Algorithm.
Tujuan optimasi adalah menurunkan error prediksi pada data pelatihan.

### 2.3 Tahap 3 - NeuroFuzzy ANN

Model ANFIS-inspired dibangun dengan PyTorch, di mana parameter membership function dan konsekuen dilatih end-to-end menggunakan optimizer Adam.

## 3. Kemampuan Aplikasi

Aplikasi dapat melakukan fungsi berikut:

1. Menerima input pengguna: usia, BMI, dan status perokok.
2. Menampilkan prediksi biaya asuransi dari tiga model (Manual FIS, GA-Tuned FIS, NeuroFuzzy ANN).
3. Menampilkan tabel komparasi hasil prediksi antar metode.
4. Menampilkan ringkasan metrik evaluasi model pada data uji (jika file metrics tersedia).

## 4. Persyaratan Sistem

### 4.1 Sistem Operasi

1. Windows 10/11, Linux, atau macOS.

### 4.2 Versi Python

1. Direkomendasikan Python 3.10 atau lebih baru.
2. Pengujian lokal terakhir dilakukan pada Python 3.12.5.

### 4.3 Library Utama

Library mengikuti file requirements.txt:

1. streamlit
2. numpy
3. pandas
4. scikit-learn
5. torch
6. joblib

## 5. Langkah Instalasi

### 5.1 Clone Repository

Jalankan perintah berikut:

git clone <url-repository>
cd Insurance-Charges-FIS-GA_Tuned-ANN

### 5.2 Buat dan Aktifkan Virtual Environment

Contoh Windows PowerShell:

python -m venv .venv
.\.venv\Scripts\Activate.ps1

### 5.3 Instal Dependensi

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

## 6. Langkah Akses Aplikasi

### 6.1 Siapkan Artifacts Model

Jika folder artifacts belum berisi file model, jalankan:

python scripts/train_and_save_artifacts.py


### 6.2 Jalankan Aplikasi Streamlit

Gunakan perintah:

python -m streamlit run apps/streamlit_app.py

Setelah berjalan, aplikasi dapat diakses melalui browser di alamat lokal Streamlit (umumnya http://localhost:8501).
