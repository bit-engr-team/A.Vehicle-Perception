# Dinamik Nesne Takip Sistemi

Bu proje, LiDAR, IMU ve kamera verilerini kullanarak otonom sürüş senaryolarında dinamik (hareketli) ve statik nesneleri tespit eden bir takip sistemidir. Python ve OpenCV, Open3D, PyTorch (YOLOv5), Kalman Filtresi gibi araçlarla geliştirilmiştir.

## 🚗 Özellikler

- **YOLOv5** ile kamera görüntüsünden nesne tespiti
- **LiDAR verisinden 3D konum çıkarımı**
- **IMU verisi ile aracın kendi hareketini hesaba katma**
- **Statik ve hareketli nesne ayrımı**
- **Takip sistemi** (UKF temelli)
- **CSV ve JSON çıktılar**
- **Test ve Canlı Mod Desteği**

## 📁 Klasör Yapısı

```
project_root/
│
├── lidar/              # .ply formatında LiDAR dosyaları
├── camera/             # .png formatında kamera görüntüleri
├── sensor_abs_data.csv # IMU verisi (timestamp, accel_x, accel_y vb.)
├── tracking.py         # Ana takip kodu
├── tracking_output.csv # Çıktı: nesne konumları ve türleri
├── tracking_output.json# Çıktı: detaylı nesne geçmişi
└── README.md           # Bu belge
```

## ⚙️ Kullanım

```bash
# Gerekli kütüphaneleri kur
pip install -r requirements.txt
```

Python dosyasındaki `MODE` değişkenini seç:

```python
MODE = "test"  # veya "live"
```

Ardından çalıştır:

```bash
python tracking.py
```

## 🔍 Giriş Verileri

- **LiDAR**: `.ply` formatında 3D nokta bulutu
- **Kamera**: `.png` formatında görüntüler
- **IMU**: `.csv` formatında; hızlanma (`accel_x`, `accel_y`), zaman damgası vb.

## 🧠 Model

- `YOLOv5x` modeli Torch Hub üzerinden yüklenir.
- Model yalnızca "confidence > 0.4" olan nesneleri işler.
- `STATIC_CLASSES` listesi üzerinden bazı sınıflar otomatik olarak statik kabul edilir.

## 💾 Çıktılar

- `tracking_output.csv`: Her karede takip edilen nesnelerin pozisyonu, hızı ve türü.
- `tracking_output.json`: Tüm kareler boyunca nesne takibi bilgisi.

## 🛠️ Bağımlılıklar

```
numpy
opencv-python
open3d
torch
pandas
scipy
filterpy
```

İsteğe bağlı: `requirements.txt` dosyası oluşturup aşağıdaki içerikle yükleyebilirsin:

```
numpy
opencv-python
open3d
torch
pandas
scipy
filterpy
```

## 📝 Notlar

- `z < 0.2` olan noktalar zemin olarak filtrelenir.
- Nesneler 3 kare üst üste hareketli ise ancak o zaman gerçekten “hareketli” sayılır (geliştirilebilir).
-

## 👨‍💻 Geliştiren

Bu sistem, otonom araç projelerinde perception (algı) modülüne destek olmak amacıyla geliştirilmiştir.
