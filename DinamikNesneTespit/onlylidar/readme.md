Aşağıda **onlylidar.py** için ayrıntılı ve parametre açıklamalı, GitHub'a uygun bir README dosyası bulabilirsin. Türkçe ve açıklamalı tuttum. Eklemek veya değiştirmek istediğin başlıkları belirtmen yeterli!

---

# LiDAR-Only Perception System

Bu proje, robotaksi ve otonom sistemlerde yalnızca **LiDAR** ve **IMU** sensör verileri kullanarak çevresel nesne tespiti, kümeleme, hareketli nesne ayrımı ve Unscented Kalman Filtresi (UKF) ile nesne takibi yapar. Kodun hem **offline/test** hem de **canlı/live** çalışma modu vardır.

## Özellikler

* LiDAR nokta bulutundan dinamik nesne ve küme tespiti
* IMU desteğiyle hız ve hareket izleme (UKF ile)
* DBSCAN tabanlı kümeleme
* Hem offline veriyle test hem de canlı ROS ortamında çalışma
* CSV ve JSON formatında çıktı
* Loglama ve hata yönetimi
* Argparse ile kolay parametre yönetimi

## Kurulum

```bash
pip install numpy open3d pandas filterpy scipy
# ROS ortamı için (live mode): rospy, ros_numpy, sensor_msgs kurulmalı
```

## Kullanım

### Test (Offline) Modu

Önceden kaydedilmiş IMU ve LiDAR verileriyle çalışır.

```bash
python onlylidar.py --mode test \
  --imu_csv <IMU_CSV_DOSYA_YOLU> \
  --lidar_folder <LIDAR_KLASORU> \
  --output_csv <CIKTI_CSV> \
  --output_json <CIKTI_JSON>
```

### Live (Canlı) Mod

Gerçek zamanlı ROS mesajları üzerinden çalışır.

```bash
python onlylidar.py --mode live \
  --output_csv <CIKTI_CSV> \
  --output_json <CIKTI_JSON>
```

> **Not:** Live modda `/imu/data` ve `/lidar/points` topic’lerinden veri dinler. ROS ortamınızın çalıştığına emin olun.

---

## Parametreler

| Parametre        | Açıklama                                     | Varsayılan Değer   |
| ---------------- | -------------------------------------------- | ------------------ |
| `--mode`         | 'test' (offline) veya 'live' (canlı)         | test               |
| `--imu_csv`      | IMU verisi CSV dosya yolu (sadece test modu) | input              |
| `--lidar_folder` | LiDAR noktalarının bulunduğu klasör (test)   | input              |
| `--output_csv`   | Sonuç CSV dosya yolu                         | output             |
| `--output_json`  | Sonuç JSON dosya yolu                        | output             |

---

## Çıktı Formatı

### CSV

| timestamp | id  | x     | y     | z     | distance |
| --------- | --- | ----- | ----- | ----- | -------- |
| float     | int | float | float | float | float    |

### JSON

```json
[
  {
    "timestamp": 1620831234.235,
    "objects": [
      {
        "id": 3,
        "real_pos": [12.8, -4.5, 0.1],
        "distance": 13.6
      },
      ...
    ]
  }
]
```

---

## Algoritma Özeti

1. **LiDAR Temizleme:** Voxel downsampling + radius outlier ile gürültü temizliği.
2. **Kümeleme:** DBSCAN ile noktalar kümelenir ve nesne merkezleri bulunur.
3. **Dinamiklik Analizi:** Frame-to-frame ICP ile hareketli noktalar belirlenir.
4. **Takip:** UKF ile her nesneye hız ve pozisyon tahmini atanır.
5. **Çıktı:** Dinamik ve statik nesneler tespit edilir, ID atanır ve kayıt edilir.

---

## Örnek Komutlar

**Test Modu:**

```bash
python onlylidar.py --mode test --imu_csv data/imu.csv --lidar_folder data/lidar --output_csv results/out.csv --output_json results/out.json
```

**Live Modu:**

```bash
python onlylidar.py --mode live --output_csv results/live.csv --output_json results/live.json
```

---

## Önemli Değişkenler ve Ayarlar

* `FRAME_RATE`: İşlenen saniyedeki frame sayısı. (default: 20)
* `MAX_MATCH_DISTANCE`: Nesne eşleştirme için maksimum mesafe. (default: 35)
* `UKF_TRACK_LIMIT`: Aynı anda izlenecek maksimum nesne sayısı. (default: 30)
* `UKF_SPEED_THRESHOLD`: Bir nesnenin dinamik (hareketli) sayılması için gereken minimum hız. (default: 0.3)
* `LIDAR_DYNAMIC_DIST_THRESHOLD`: Hareketli nokta ayrımı için eşik. (default: 0.2)
* `MOTION_WINDOW_SIZE`, `MOTION_COUNT_THRESHOLD`: Dinamiklik kararında pencere boyutu ve eşik.

---

## Klasör/Veri Yapısı Tavsiyesi

```
data/
 ├─ lidar/         # .ply LiDAR frame dosyaları
 └─ imu.csv        # IMU zaman serisi verisi (timestamp, accel_x, accel_y ...)
results/
 ├─ lidar.csv      # Sonuç CSV
 └─ lidar.json     # Sonuç JSON
```

---

## Lisans

MIT

---

## İletişim

Herhangi bir soru için: \[github profilin veya e-posta]

---

**Not:** Kodun ve README'nin güncel kalmasını sağlamak için parametreleri ve yol isimlerini kendi verinize göre güncelleyin.

---

Eklemek istediğin başka detay olursa tekrar düzenleyebilirim!
