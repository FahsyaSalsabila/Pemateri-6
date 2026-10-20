# Studi Kasus: Identifikasi Penyebab Deforestasi Menggunakan Citra Satelit dan AI

## 1. Latar Belakang

Sebuah kawasan hutan tropis di Indonesia terdeteksi mengalami kehilangan tutupan pohon berdasarkan *deforestation alerts*. Peringatan tersebut menunjukkan lokasi terjadinya perubahan hutan, tetapi belum menjelaskan penyebab perubahan tersebut.

Tim pemantauan hutan ingin mengetahui apakah gangguan yang terdeteksi berkaitan dengan pembukaan lahan pertanian, pembangunan jalan, penebangan selektif, atau pertambangan.

Untuk membantu proses identifikasi, dikembangkan sistem berbasis *Artificial Intelligence* (AI) menggunakan *Convolutional Neural Network* (CNN). Sistem ini dirancang untuk mempelajari pola spasial pada citra satelit dan mengklasifikasikan kemungkinan penyebab langsung gangguan hutan.

## 2. Identifikasi Masalah

Permasalahan utama dalam studi kasus ini adalah:

1. *Deforestation alerts* hanya menunjukkan lokasi perubahan tutupan pohon.
2. Penyebab gangguan hutan tidak selalu dapat diketahui dari informasi peringatan saja.
3. Pemeriksaan manual terhadap banyak lokasi membutuhkan waktu dan sumber daya.
4. Diperlukan metode otomatis untuk membantu mengelompokkan kemungkinan penyebab gangguan hutan.

**Pertanyaan penelitian:**

Bagaimana CNN dapat digunakan untuk mengklasifikasikan kemungkinan penyebab langsung gangguan tutupan hutan berdasarkan pola citra satelit?

## 3. Data dan Variabel Penelitian

Pada implementasi awal, digunakan dataset sintetis untuk menguji alur kerja CNN.

### 3.1 Spesifikasi Dataset

| Komponen | Keterangan |
|---|---|
| Data masukan | Patch simulasi citra multi-kanal |
| Ukuran citra | 32 × 32 piksel |
| Kanal demonstrasi | VV, VH, NIR, dan RED |
| Jumlah sampel | 800 patch sintetis |
| Jumlah kelas | 5 kelas penyebab gangguan |
| Pembagian data | 80% pelatihan dan 20% pengujian |
| Model | Convolutional Neural Network (CNN) |
| Evaluasi | Accuracy, precision, recall, F1-score, dan confusion matrix |

### 3.2 Kategori Penyebab Deforestasi

Model dirancang untuk mengklasifikasikan gangguan hutan ke dalam lima kategori berikut:

1. Pertanian skala kecil
2. Pertanian skala besar
3. Pembangunan jalan
4. Penebangan selektif
5. Pertambangan

**Catatan:** Dataset yang digunakan merupakan data sintetis untuk demonstrasi kode, bukan data observasi hutan Indonesia atau pengukuran Sentinel yang sebenarnya. Kanal VV dan VH umumnya berkaitan dengan citra radar, sedangkan NIR dan RED berkaitan dengan kanal optik. Dalam penerapan nyata, data harus disiapkan dan diselaraskan secara tepat sebelum digunakan sebagai masukan model.

## 4. Implementasi Penyelesaian

Implementasi sistem dilakukan melalui lima tahapan utama.

### 4.1 Pembuatan Dataset

Membuat 800 patch citra sintetis berukuran 32 × 32 piksel dengan empat kanal masukan. Setiap patch diberikan label sesuai dengan salah satu dari lima kategori penyebab gangguan hutan.

### 4.2 Preprocessing Data

Melakukan normalisasi nilai piksel, menyiapkan label kelas, dan mengubah data ke format tensor agar dapat diproses oleh model CNN.

### 4.3 Pelatihan Model CNN

Melatih model CNN menggunakan data pelatihan agar dapat mempelajari pola spasial yang membedakan setiap kategori gangguan hutan.

### 4.4 Pengujian Model

Menguji model menggunakan data yang tidak digunakan selama pelatihan. Hasil prediksi kemudian dibandingkan dengan label sebenarnya untuk mengukur kinerja model.

### 4.5 Evaluasi dan Analisis

Mengevaluasi kinerja model menggunakan accuracy, precision, recall, F1-score, dan confusion matrix. Hasil evaluasi digunakan untuk mengetahui kemampuan model dalam mengenali setiap kategori serta mengidentifikasi kelas yang sering tertukar.

### 4.6 Contoh Menjalankan Program

Setelah seluruh file implementasi tersedia dan dependensi terpasang, program dapat dijalankan menggunakan perintah berikut:

```bash
python generate_demo_data.py
python -m src.train --epochs 8
python predict.py
```

Program dirancang untuk menghasilkan beberapa keluaran:

- `models/deforestation_cnn.pt` — model CNN yang telah dilatih.
- `outputs/metrics.json` — hasil evaluasi model dalam format JSON.
- `outputs/confusion_matrix.png` — visualisasi hasil klasifikasi.

**Catatan:** Perintah dan nama keluaran di atas harus disesuaikan dengan struktur serta nama file implementasi yang benar-benar tersedia di repository.

## 5. Analisis Hasil

Hasil pengujian dianalisis menggunakan beberapa metrik evaluasi berikut.

| Metrik | Fungsi |
|---|---|
| Accuracy | Mengukur proporsi seluruh prediksi yang benar. |
| Precision | Mengukur ketepatan prediksi pada suatu kelas. |
| Recall | Mengukur kemampuan model menemukan sampel dari suatu kelas. |
| F1-score | Menggabungkan precision dan recall menjadi satu ukuran. |
| Confusion matrix | Menunjukkan jumlah prediksi benar dan kesalahan klasifikasi antarkelas. |

Nilai aktual setiap metrik harus diperoleh dari hasil eksekusi program dan dicantumkan dalam laporan setelah pengujian selesai.

Karena dataset bersifat sintetis, hasil evaluasi hanya menggambarkan kinerja model pada data simulasi. Hasil tersebut tidak dapat langsung dianggap sebagai tingkat akurasi untuk mengidentifikasi penyebab deforestasi di dunia nyata.

## 6. Pembahasan

CNN berpotensi membantu proses klasifikasi karena mampu mempelajari bentuk, pola, dan distribusi spasial pada citra. Namun, pola yang serupa dapat muncul dari aktivitas yang berbeda. Sebagai contoh, pembukaan lahan untuk pertanian dan pertambangan sama-sama dapat menghasilkan area terbuka.

Dalam penerapan nyata, model perlu menggunakan citra Sentinel-1 dan Sentinel-2 yang sesuai dengan waktu terjadinya peringatan deforestasi. Label penyebab gangguan juga perlu diverifikasi menggunakan citra resolusi tinggi, data pendukung, atau observasi lapangan.

Selain itu, pengujian sebaiknya dilakukan pada lokasi yang tidak digunakan selama pelatihan untuk mengukur kemampuan model dalam mengenali pola di kawasan hutan yang berbeda. Langkah ini membantu mengurangi risiko memperoleh hasil evaluasi yang terlalu optimistis.

## 7. Kesimpulan

Studi kasus ini mendemonstrasikan rancangan penggunaan CNN untuk mengklasifikasikan kemungkinan penyebab langsung gangguan tutupan hutan ke dalam lima kategori, yaitu pertanian skala kecil, pertanian skala besar, pembangunan jalan, penebangan selektif, dan pertambangan.

Implementasi mencakup pembuatan dataset sintetis, preprocessing, pelatihan model, pengujian prediksi, serta evaluasi kinerja. Pendekatan ini dapat menjadi dasar pengembangan sistem pemantauan deforestasi yang lebih efisien.

Namun, penerapan untuk pengambilan keputusan lingkungan membutuhkan data satelit nyata, label yang terverifikasi, evaluasi independen, dan validasi lapangan.


- **Deforestation Alerts:** Peringatan yang menunjukkan indikasi perubahan atau kehilangan tutupan pohon.

