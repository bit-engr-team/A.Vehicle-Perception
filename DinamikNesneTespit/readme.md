 dynamicdetection2 
 ---livemode.py ( live mod )
 --- dynamicdetection2.py (ror kullanılan yeni versiyon)


---

# Robotaxi Perception System

Bu proje, **LiDAR**, **IMU** ve **kamera** verilerini kullanarak hareketli ve statik nesneleri algılayan, takip eden ve sınıflandıran bir ROS 2 uyumlu **Perception Sistemi** içerir. Kod hem **test modunda** hem de **live modunda** çalışacak şekilde tasarlanmıştır.

---

## Genel Bakış

* **Test Modu:** Kayıtlı sensör verileri üzerinde çalışır. LiDAR, IMU ve kamera kayıtlarını senkronize ederek nesne tespiti, takip ve sınıflandırma yapar. Sonuçları CSV ve JSON dosyalarına yazar. Ayrıca görüntü üzerinde tespitleri görselleştirir.

* **Live Modu:** Gerçek zamanlı olarak ROS 2 sensör topiclerinden veri alır (IMU, LiDAR, kamera). Nesneleri algılar ve takip eder, çıktıyı dosyaya kaydeder. Sürekli çalışır, elle durdurulana kadar açık kalır.

---

## Kullanım

```bash
python robotaxi_perception_final.py --mode test
# veya
python robotaxi_perception_final.py --mode live
```

---

## Modlar Detayları

### Test Modu

* **Girdi:**

  * IMU verisi: CSV dosyası (örnek: `sensor_abs_data.csv`), içinde `timestamp`, `accel_x`, `accel_y` gibi sütunlar var.
  * LiDAR verileri: PLY formatında point cloud dosyaları (`lidar` klasörü).
  * Kamera görüntüleri: PNG formatında (`camera` klasörü).

* **İşlem:**

  * Her frame için kamera görüntüsü üzerinde YOLOv5 ile nesne algılama yapılır.
  * Algılanan nesnelerin görüntü koordinatları ile LiDAR noktaları eşleştirilir (en yakın nokta bulunur).
  * IMU verisinden araç ivmesi hesaplanır, bu ivme nesnelerin hızını tahmin etmekte kullanılır.
  * Unscented Kalman Filter (UKF) ile nesne pozisyonları takip edilir.
  * Statik ve dinamik nesneler hız eşiklerine göre sınıflandırılır.
  * Sonuçlar hem CSV hem JSON olarak kaydedilir.
  * Kamera görüntüsünde tespitler gösterilir.

* **Çıktı:**

  * `output/test_tracking_output.csv`
  * `output/test_tracking_output.json`

### Live Modu

* **Girdi:**

  * ROS 2 üzerinden gerçek zamanlı sensör topicleri:

    * `/imu/data` (IMU)
    * `/velodyne_points` (LiDAR)
    * `/camera/image_raw` (Kamera)

* **İşlem:**

  * Test modundaki işleyişin aynısı, ancak sensör verileri gerçek zamanlı olarak alınır.
  * UKF ile nesne pozisyonları takip edilir ve hareketli/durgun ayrımı yapılır.
  * Her yeni kamera görüntüsünde nesne takibi güncellenir.
  * Çıktılar anlık CSV dosyasına kaydedilir.

* **Çıktı:**

  * `output/live_output.csv`

---

## Önemli Parametreler

| Parametre            | Açıklama                                     | Optimal Değer | Değer Artarsa                            | Değer Azalırsa                                       |
| -------------------- | -------------------------------------------- | ------------- | ---------------------------------------- | ---------------------------------------------------- |
| `FRAME_RATE`         | Sistem kare/saniye oranı                     | 20 FPS        | Daha hızlı ama işlem yükü artar          | Daha yavaş, gecikme olabilir                         |
| `MAX_MATCH_DISTANCE` | Takip için eşleştirme mesafe eşiği (m)       | 2.5 m         | Çok büyük olursa yanlış eşleşmeler artar | Çok küçük olursa takip zorlaşır, yeni ID’ler çoğalır |
| `model.conf`         | YOLOv5 algılama güven eşiği                  | 0.3           | Düşükse yanlış pozitif artar             | Çok yüksekse gerçek nesneler gözükmeyebilir          |
| `UKF_TRACK_LIMIT`    | Aynı anda takip edilen maksimum nesne sayısı | 30 nesne      | Bellek ve işlem yükü artar               | Takip kaybı ve algılamada yetersizlik                |

---

## Kodun Temel İşleyişi

* **Nesne algılama:** Kamera görüntüsüne YOLOv5 uygulanır.
* **LiDAR temizleme:** Zemin altı noktalar çıkarılır, gürültü azaltılır.
* **Nesne-LiDAR eşleştirme:** Görüntüdeki nesneler ile LiDAR noktaları mesafeye göre eşleştirilir.
* **Hareketlilik tahmini:** IMU’dan araç ivmesi alınır, nesnelerin hız tahmini yapılır.
* **Takip:** Unscented Kalman Filter kullanılarak nesne pozisyonları stabilize edilir.
* **Sınıflandırma:** Statik nesneler listesi ve hız eşiklerine göre hareketli/ statik ayrımı yapılır.
* **Çıktı kaydı:** Zaman damgası, nesne pozisyonu, mesafe ve hareketlilik türü CSV ve JSON’a yazılır.

---

## Gereksinimler

* Python 3.7+
* PyTorch (YOLOv5 için)
* Open3D
* OpenCV
* FilterPy
* SciPy
* Pandas
* ROS 2 (live mod için)
* CvBridge, sensor\_msgs, rclpy (live mod için)

---

## Notlar

* `STATIC_CLASSES` listesine eklenen nesneler her zaman statik kabul edilir (örneğin "kite" etiketi).
* Kod, LiDAR noktasının gerçek konumunu doğrudan kullanır; aracın hareketi IMU ivmesiyle yaklaşık olarak tahmin edilir.
* Test modu kolayca kendi verilerinizle deneyip çıktı almanızı sağlar.
* Live modda gerçek sensör verileri ile ROS 2 altyapısında çalışır.

---

## İletişim

Kodla ilgili soruların veya önerilerin için bana ulaşabilirsin. Kolay gelsin! 🚗🤖
