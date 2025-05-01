## LİVE MODE EKSİK (TAMAMLANMADI)
 # 🚗 Sensor Fusion ile Hareketli Nesne Takibi

Bu proje, LiDAR, IMU ve kamera verilerini birleştirerek gerçek zamanlı nesne tespiti, takip ve hareketli/statik ayrımı yapan bir sistem sunar. YOLOv5 modeli ile görsel nesne tespiti, Kalman Filtresi ile konum ve hız takibi, ve IMU ivmesi ile bağıl hareket analizi birleştirilmiştir.

---

## 🔧 Girdiler

### 1. IMU Verisi (`sensor_abs_data.csv`)
CSV formatında aşağıdaki sütunlara sahiptir:
- `timestamp`: UNIX zaman damgası (float)
- `abs-latitude`, `abs-longitude`, `abs-altitude`: GPS konumu (isteğe bağlı)
- `accel_x`, `accel_y`, `accel_z`: İvme değerleri (m/s²)
- `gyro_x`, `gyro_y`, `gyro_z`: Açısal hız (rad/s)

### 2. LiDAR Verisi (`lidar/`)
- Her kare için `.ply` formatında nokta bulutu dosyası
- Dosya ismi zaman bilgisi içerir: `lidar_1680001234.456.ply`

### 3. Kamera Görüntüleri (`camera/`)
- Her kare için `.png` formatında RGB görüntüler
- Dosya ismi sıralı olmalıdır: `00001.png`, `00002.png`, ...

---

## 📤 Çıktılar

### `tracking_output.csv`
Aşağıdaki kolonları içerir:
| Kolon Adı       | Açıklama                                |
|----------------|------------------------------------------|
| object_id       | Takip edilen nesne ID’si                |
| frame           | Kare numarası                           |
| x, y            | Nesnenin dünya koordinatları (metre)    |
| vx, vy          | Nesnenin x ve y yönündeki hızları       |
| ax, ay          | İvme (şu an sabit `0`, genişletilebilir)|
| distance        | Araçtan olan mesafe (metre)             |
| type            | `Static` veya `Dynamic`                 |
| class           | YOLO sınıf adı (örneğin: `car`, `person`) |

### `tracking_output.json`
Her nesne ve kare için aşağıdaki bilgileri içerir:
{
  "frame": 1,
  "id": 3,
  "class": "car",
  "type": "Dynamic",
  "position": [12.4, 5.7],
  "velocity": [0.8, -0.1],
  "distance": 13.2
}

### Gerçek Zamanlı Görüntüleme
- Her nesne için sınıf, ID, pozisyon ve hız ekranda gösterilir.
- Renk kodları:
  - 🟥 Kırmızı kutu: Statik nesne
  - 🟩 Yeşil kutu: Dinamik nesne

---

## 🧠 Yöntem Özeti

- **YOLOv5 (n)**: Görüntüdeki nesneleri algılar (CPU modunda çalışır)
- **LiDAR + Projeksiyon**: Gerçek dünya konumu hesaplanır
- **IMU ivmesi**: Aracın kendi hareketi tahmin edilir
- **Unscented Kalman Filter (UKF)**: Nesnelerin pozisyon ve hız tahmini yapılır
- **Bağıl Hız Analizi**: Nesne aracın hızından bağımsız hareket ediyorsa “Dynamic” kabul edilir

---

## ⚙️ Kurulum

```bash
pip install -r requirements.txt
```

`requirements.txt` içeriği:
```
torch
opencv-python
numpy
open3d
pandas
filterpy
scipy
```

---

## ▶️ Çalıştırma

Python dosyasını doğrudan çalıştırın:

```bash
python main.py
```

Klavye kontrolü:
- `q`: Görüntüleme penceresini kapat ve işlemi sonlandır

---

## 📁 Dosya Yapısı Örneği

```
project_root/
│
├── main.py
├── sensor_abs_data.csv
├── lidar/
│   ├── lidar_1680001234.456.ply
│   └── ...
├── camera/
│   ├── 00001.png
│   └── ...
└── tracking_output.csv / tracking_output.json
```

---

## 📝 Notlar

- Statik sınıflar listesi `STATIC_CLASSES` içinde tanımlıdır. Gerekirse genişletilebilir.
- IMU verisi eksik karelerde takip yapılmaz.
- YOLO tahmini ile zemin nesneleri (örneğin trafik ışığı, bank, duvar) varsayılan olarak “Statik” kabul edilir.
- İvme verisi filtrelenmemiştir. Daha hassas analiz için Kalman ya da düşük geçiren filtre eklenebilir.

---

## ✨ Geliştirme Fikirleri

- Kamera-LiDAR kalibrasyonu ile daha hassas projeksiyon
- IMU/GNSS verisiyle global konum haritalama
- Hız, ivme filtreleme ve eğik yüzeylerde hareket analizi
- Otomatik ground-truth üretimi ve doğruluk metriği hesaplama

---

## 🧑‍💻 Yazar

**[Ad Soyadınızı Buraya Yazabilirsiniz]**  
Lütfen kaynak göstererek kullanınız.
