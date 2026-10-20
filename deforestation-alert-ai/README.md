# Attributing Direct Drivers to Deforestation Alerts with Satellite Imagery and AI

Studi kasus edukatif: klasifikasi penyebab langsung gangguan tutupan hutan dari patch citra satelit menggunakan CNN.

## Tujuan
Membangun alur kerja sederhana untuk:
1. Membuat data demo citra multi-kanal (simulasi, bukan citra satelit asli).
2. Melatih CNN untuk mengklasifikasikan 5 kelas: `smallholder_agriculture`, `large_scale_agriculture`, `road`, `selective_logging`, dan `mining`.
3. Mengevaluasi model dan menyimpan bobot serta metrik.
4. Menyiapkan jalur untuk mengganti data demo dengan patch Sentinel-1/Sentinel-2 berlabel.

> **Penting:** data demo dibuat secara sintetis agar proyek dapat dijalankan langsung. Akurasi pada data demo tidak menunjukkan performa di dunia nyata. Untuk penelitian sebenarnya, gunakan data satelit dan label yang divalidasi oleh ahli/lapangan.

## Struktur folder
```text
deforestation-alert-ai/
├── data/                  # data demo dibuat otomatis
├── models/                # bobot model tersimpan di sini
├── outputs/               # metrik dan confusion matrix
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── generate_demo_data.py
├── predict.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 1. Buka di VS Code
Ekstrak ZIP, lalu buka folder `deforestation-alert-ai` melalui **File → Open Folder** di VS Code.

## 2. Buat environment dan install dependency
Python 3.10 atau 3.11 direkomendasikan.

Windows PowerShell:
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Jika PowerShell memblokir aktivasi, jalankan perintah berikut atau gunakan terminal Command Prompt:
```powershell
.\.venv\Scripts\activate.bat
```

## 3. Jalankan eksperimen demo
```bash
python generate_demo_data.py
python -m src.train --epochs 8
```
Hasil:
- `data/demo_patches.npz`: patch sintetis 4-kanal dan label.
- `models/deforestation_cnn.pt`: bobot model.
- `outputs/metrics.json`: akurasi dan laporan klasifikasi.
- `outputs/confusion_matrix.png`: confusion matrix.

## 4. Prediksi satu patch
Setelah training:
```bash
python predict.py
```
Perintah ini mengambil satu patch dari data demo dan menampilkan kelas prediksi beserta probabilitasnya.

## Metodologi studi kasus
- **Masalah:** sebuah sistem alert mendeteksi gangguan/kehilangan tutupan pohon, tetapi tidak mengidentifikasi penyebab langsungnya.
- **Input model:** patch citra 32×32 dengan empat kanal demonstrasi: `VV`, `VH`, `NIR`, `RED`. Dalam penggunaan sebenarnya, definisikan kanal dan normalisasi secara konsisten. Sentinel-1 menyediakan kanal radar seperti VV/VH (tergantung produk); Sentinel-2 menyediakan kanal optik seperti merah dan NIR.
- **Label:** penyebab langsung yang diverifikasi, misalnya pertanian skala kecil/besar, jalan, penebangan selektif, dan pertambangan.
- **Model:** CNN 2D.
- **Evaluasi:** akurasi, precision/recall/F1 per kelas, dan confusion matrix.
- **Risiko:** kelas tidak seimbang, label bising, perbedaan wilayah/musim, awan pada citra optik, serta kebingungan antara aktivitas manusia dan gangguan alami.

## Mengganti data sintetis dengan data sungguhan
1. Kumpulkan alert deforestasi dan citra Sentinel-1/Sentinel-2 yang berdekatan secara waktu dengan alert.
2. Tentukan jendela patch dan kanal yang konsisten. Lakukan masking awan untuk Sentinel-2 dan normalisasi data radar/optik.
3. Beri label penyebab berdasarkan interpretasi citra resolusi tinggi, data lapangan, atau sumber tepercaya.
4. Simpan `X` sebagai array float berbentuk `(jumlah_patch, tinggi, lebar, kanal)` dan `y` sebagai integer kelas `0..4`, dalam file `.npz` dengan kunci `X` dan `y`. Lihat `src/dataset.py`.
5. Pisahkan data berdasarkan lokasi/waktu, bukan hanya acak patch, untuk mengurangi kebocoran spasial.
6. Latih ulang dan evaluasi pada lokasi yang tidak digunakan saat training.

Contoh kelas (urutan harus sama dengan kode):
```text
0 smallholder_agriculture
1 large_scale_agriculture
2 road
3 selective_logging
4 mining
```

## Catatan Sentinel / Google Earth Engine
Proyek ini **tidak mengunduh citra Sentinel secara otomatis** dan tidak membutuhkan akun cloud untuk demo. Untuk data nyata, Google Earth Engine dapat dipakai untuk menyeleksi Sentinel-1/Sentinel-2, masking awan, membuat komposit, lalu mengekspor patch. Jangan memasukkan kredensial atau token ke GitHub. Tambahkan metadata waktu, lokasi, sumber label, dan versi preprocessing agar eksperimen dapat direproduksi.

## Upload ke GitHub
1. Buat repository baru di GitHub, misalnya `deforestation-alert-ai`.
2. Buka terminal di folder proyek:
```bash
git init
git add .
git commit -m "Add deforestation alert driver classification case study"
git branch -M main
git remote add origin https://github.com/USERNAME/deforestation-alert-ai.git
git push -u origin main
```
Ganti `USERNAME` dengan username GitHub Anda. Jangan upload `.venv/`, file rahasia, atau dataset satelit berlisensi/terbatas. Folder `data/` demo dapat dihasilkan ulang dengan `python generate_demo_data.py`.

## Batasan ilmiah
Model hanya memprediksi kelas dari pola input. Prediksi bukan bukti kausal dan tidak menggantikan verifikasi lapangan. Sistem operasional perlu validasi independen, ambang confidence, kelas `unknown/other`, dan pemantauan perubahan performa antarwilayah.
