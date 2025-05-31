# Robotaxi Perception Sistemi

Bu proje, otonom araçlar için gerçek zamanlı nesne algılama, takip ve sınıflandırma gerçekleştiren bir algılama sistemidir. Kamera, LiDAR ve IMU sensör verilerini kullanarak dinamik ve statik nesneleri algılar ve konumlandırır.



# Robotaxi Perception Sistemi - Kod Versiyon Farkları

Bu doküman, robotaxi algılama sistemine ait iki kod versiyonunun temel farklılıklarını özetler.

| Özellik                                | Versiyon 1 (Basit)             | Versiyon 2 (Gelişmiş)                                                    |
| -------------------------------------- | ------------------------------ | ------------------------------------------------------------------------ |
| **LiDAR temizleme yöntemi**            | Statistical Outlier Removal    | Radius Outlier Removal ve Voxel Downsampling                             |
| **UKF Kalman filtresi parametreleri**  | Q ve R matrisleri basit ayarlı | Q ve R matrisleri optimize edilmiş (hareket hassasiyeti artırılmış)      |
| **Dinamik nesne belirleme kriterleri** | Yalnızca hız tabanlı           | Optik akış, UKF hızı ve LiDAR dinamik kümelemesine dayalı (çok kriterli) |
| **YOLO model güven eşik değeri**       | Güven eşik değeri düşük (0.3)  | Güven eşik değeri artırılmış (0.4)                                       |
| **Görüntü-LiDAR koordinat dönüşümü**   | Basit projeksiyon yöntemi      | k-NN ortalaması ile gerçek pozisyon daha hassas belirleniyor             |
| **Logging ve hata yönetimi**           | Yok veya minimal               | Kapsamlı logging ve hata yakalama mekanizmaları eklendi                  |
| **Optik akış analizi**                 | Kullanılmıyor                  | OpenCV Farneback yöntemi ile optik akış analizi eklenmiş                 |
| **LiDAR Dinamik kümeleme**             | Kullanılmıyor                  | ICP ve DBSCAN algoritmalarıyla LiDAR hareket analizi yapılır             |
| **Parametre Ayarlanabilirliği**        | Sabit parametreler             | Argparse modülüyle dinamik ve esnek parametre yönetimi                   |
| **Kod yapısı ve okunabilirlik**        | Tek düze yapı, az yorum satırı | Modüler fonksiyonlar, açıklayıcı yorumlar ve daha okunabilir yapı        |

Bu tablo, her iki versiyon arasındaki ana farklılıkları ve versiyon 2’nin algoritma karmaşıklığını ve sistem doğruluğunu artırmak amacıyla eklenen gelişmiş özellikleri göstermektedir.




---

## Kullanılan Algoritmalar

 Algoritma                      Açıklama                                       
 -----------------------------  ---------------------------------------------- 
 YOLOv5 (yolov5n)               Görüntü tabanlı nesne algılama.                
 Unscented Kalman Filter (UKF)  Nesnelerin konum ve hız takibi.                
 Optik Akış (Farneback)         Kamera görüntüsünde hareket algılama.          
 ICP (Point Cloud Alignment)    LiDAR noktalarını hizalama ve hareket tespiti. 
 DBSCAN                         LiDAR noktalarını kümeleme.                    
 Hungarian Algorithm            Nesne takip ve eşleştirme.                     

---

## Girdi Verileri

 Veri Türü  Test Modu Girdileri                              Canlı Mod Girdileri       
 ---------  -----------------------------------------------  ------------------------- 
 Kamera     PNG görüntüleri (klasörden okunur)               ROS `camera_image` topic 
 LiDAR      PLY formatında nokta bulutları                   ROS `lidar_points` topic 
 IMU        CSV dosyası (`timestamp`, `accel_x`, `accel_y`)  ROS `imu` topic          

---

## Çıktılar

 Dosya Türü  İçerik                                                            
 ----------  ----------------------------------------------------------------- 
 CSV         `timestamp, id, x, y, z, distance` bilgileri içerir               
 JSON        Nesne kimliği, gerçek konumu ve mesafesi                          
 Görüntü     Algılanan nesneler, kutular ve hareket oklarıyla görselleştirilir 

---

## Temel Parametreler

 Parametre                        Varsayılan Değer  Açıklama                                                                   
 -------------------------------  ----------------  -------------------------------------------------------------------------- 
 FRAME_RATE                      20                İşlenecek kare sayısı (FPS).                                               
 MAX_MATCH_DISTANCE             35                Takip edilen ve yeni nesneler arasındaki maksimum eşleşme mesafesi.        
 UKF_TRACK_LIMIT                30                Eşzamanlı takip edilebilecek maksimum nesne sayısı.                        
 UKF_SPEED_THRESHOLD            0.3               Nesnenin dinamik olarak kabul edilmesi için minimum hız eşik değeri (ms). 
 MOTION_COUNT_THRESHOLD         3                 Hareketli kabul etmek için gereken hareketli kare sayısı.                  
 FLOW_THRESHOLD                  1.0               Optik akışta hareket tespiti için eşik değer (pikselframe).               
 LIDAR_DYNAMIC_DIST_THRESHOLD  0.2               LiDAR noktalarında hareket tespiti için mesafe eşik değeri (metre).        
 LIDAR_CLUSTER_TRACK_DIST      2.0               LiDAR küme merkezlerinin takibe atanması için maksimum uzaklık (metre).    

---

## Kurulum ve Çalıştırma

### Bağımlılıklar

```bash
pip install opencv-python open3d torch pandas scipy filterpy rospy ros_numpy cv_bridge
```

### Çalıştırma

Test Modu

```bash
python3 perception.py --mode test --imu_csv pathtoimu.csv --lidar_folder pathtolidar --camera_folder pathtocamera --output_csv output.csv --output_json output.json
```

Canlı Mod (ROS)

```bash
rosrun robotaxi_perception perception.py --mode live --output_csv live_output.csv --output_json live_output.json
```

---



