#!/usr/bin/env python3
from ultralytics import YOLO

def main():
    # 1) En iyi ağırlıkları al
    best_weights = "runs/train/weights/best.pt"
    model = YOLO(best_weights)

    # 2) Tahmini yap ve kaydet
    results = model.predict(
        source="../dataset/test/images/",  # tek resim de olur, klasör de
        conf=0.25,                         # minimum güven eşiği
        device="0",                        # GPU 0, CPU için "cpu"
        save=True,                         # sonuçları kaydet
        save_dir="runs/detect"            # runs/detect altında
    )

    # 3) Sonucu bildir
    print("✅ Inference tamamlandı!")
    print("   • Çıktılar:", results.path)

if __name__ == "__main__":
    main()