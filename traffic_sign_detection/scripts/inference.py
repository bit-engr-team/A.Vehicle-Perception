from ultralytics import YOLO
import json
from pathlib import Path
from datetime import datetime

def main():
    # —————————————————————————————————————
    # 1) Ayarlar
    # —————————————————————————————————————
    WEIGHTS       = "best.pt"
    SOURCE_DIR    = Path("test/images") # Test edilecek görsellerin bulunduğu adres
    OUTPUT_DIR    = Path("runs/json_results")
    REF_PX_WIDTH  = 100  # 1× mesafedeki işaretin piksel genişliği(Uzaklığını bildiğimiz bir tabelayı kullanarak gerçek değeri hesaplayacağız. 
    #Şu anlık hangi trafik işareti daha yakın bilmemiz için yeterli)

    # Modeli yükle, sınıf isimlerini al
    model       = YOLO(WEIGHTS)
    class_names = model.names

    # Çıktı klasörünü hazırla
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # —————————————————————————————————————
    # 2) Stream modda tahmin (görsel kaydetme kapalı)
    # —————————————————————————————————————
    for r in model.predict(
            source=str(SOURCE_DIR),
            conf=0.25,
            device="cpu", # şu anlık cpu ile tahmin açık
            save=False,
            verbose=False,
            stream=True
        ):
        # 3) Anlık JSON üretimi
        timestamp  = datetime.now().isoformat()
        img_name   = Path(r.path).name

        predictions = []
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            px_width      = x2 - x1
            ratio_x       = REF_PX_WIDTH / px_width if px_width > 0 else None

            predictions.append({
                "class":          class_names[int(box.cls[0])],
                "conf":           round(float(box.conf[0]), 4),
                "bbox":           [round(x, 2) for x in (x1, y1, x2, y2)],
                "distance_ratio": round(ratio_x, 2) if ratio_x else None
            })

        result_obj = {
            "timestamp":   timestamp,
            "image":       img_name,
            "predictions": predictions
        }

        # JSON'u kaydet
        json_path = OUTPUT_DIR / f"{img_name}.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(result_obj, f, indent=2, ensure_ascii=False)

        # Konsola anlık yazdır (isteğe bağlı)
        print(json.dumps(result_obj, ensure_ascii=False))

    print("✅ Tüm görüntüler işlendi ve sadece JSON üretildi.")
    print(f"   • JSON dizini: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
