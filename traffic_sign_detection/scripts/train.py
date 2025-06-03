#!/usr/bin/env python3
from ultralytics import YOLO

def main():
    # 1) Modeli yükle (önceden eğitilmiş ağırlık)
    model = YOLO("yolo11l.pt")

    # 2) Eğitimi çalıştır
    results = model.train(
        data="config/data.yaml",  # data.yaml dosyanızın yolu
        imgsz=640,                   # resim boyutu
        epochs=24,                   # epoch sayısı
        batch=16,                     # batch boyutu
        device="0",                  # GPU 0, CPU için "cpu"
        workers=8,                   # işçi sayısı
        project="runs",           # çıktıların kaydolacağı ana klasör
        name="train",                # runs/train alt klasörü
        exist_ok=True                # üzerine yazılsın
    )

    # 3) Bittiğinde bilgi ver
    print("✅ Eğitim tamamlandı!")
    print("   • En iyi ağırlık:", results.best)
    print("   • Çıktı klasörü:", results.path)

if __name__ == "__main__":
    main()
