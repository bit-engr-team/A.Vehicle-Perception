# YOLOv11 Traffic Sign Detection

**Bu repo**, YOLOv11 ile Türkiye trafik levhalarını tespit eden bir derin öğrenme projesi içerir.

---

## 📋 İçindekiler
1. [Proje Açıklaması](#proje-açıklaması)  
2. [Kurulum](#kurulum)  
3. [Veri Seti](#veri-seti)  
4. [Eğitim](#eğitim)  
5. [Çıkarım (Inference)](#çıkarım-inference)  
6. [Dosya Yapısı](#dosya-yapısı)  

---

## 🔍 Proje Açıklaması
Türkiye’deki trafik levhalarını tespit etmek için Ultralytics YOLOv11 modelini kullanan bir örnek uygulama.  
- Eğitim: Kendi etiketli veri setinizle model eğitimi  
- Çıkarım: Yeni görüntülerde nesne tespiti  

---

## 🛠 Kurulum

1. Repo’yu klonlayın  
   ```bash
   git clone https://github.com/<KullaniciAdin>/yolo11-traffic-sign.git
   cd yolo11-traffic-sign