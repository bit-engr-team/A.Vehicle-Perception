# SeritTespit.py - Real-Time Image Processing for Autonomous Driving

## Overview
Kısa özet:
# SOURCE ARGÜMANIYLA ALDIĞI FOLDERA EKLENEN YENİ FOTOĞRAFLARI İŞLİYOR SONUÇLARI runs/exp/detections.jsona CANLI OLARAK YAZIYOR
# KABUL ETTİĞİ GÖRÜNTÜNÜN DIMENSIONINA GÖRE OLAN AYAR SIKINTILI!!! DÜZELTİLECEK
# ŞUANLIK ESCAPE BASINCA FOLDER'I İZLEMEYİ BIRAKIYOR

`SeritTespit.py` is a Python script designed for real-time perception tasks in autonomous driving. It uses the YOLOPv2 model to perform:
- **Object Detection**: Identifies objects (e.g., cars, pedestrians, trucks) with bounding boxes and confidence scores.
- **Lane Line Detection**: Detects and smooths lane lines, computing centerlines.
- **Drivable Area Segmentation**: Identifies the drivable area with boundary and mesh points.
- **Bird's-Eye View (BEV) Transformation**: Transforms images and masks to a top-down view for better spatial understanding.

The script monitors a specified folder for new images (`.jpg` or `.png`), processes them as they are added, and continues running until the Escape key is pressed. It supports visualization of results and saves processed images, BEV images, and detection data in JSON format.

## Requirements
- **Python**: 3.8 or higher
- **Dependencies**:
  ```bash
  pip install torch opencv-python numpy matplotlib
  ```
- **Hardware**:
  - Apple Silicon (MPS) or CPU (CUDA not supported in this version).
  - Sufficient memory for real-time image processing.
- **Model Weights**:
  - YOLOPv2 model weights (`yolopv2.pt`), expected in `data/weights/`.
  - Download or train the model as needed (not provided in this repository).
- **Utilities**:
  - The script requires a `utils.utils` module with functions like `time_synchronized`, `non_max_suppression`, etc. Ensure this module is available in the project directory.

## Installation
1. Clone or download the repository containing `SeritTespit.py`.
2. Place the YOLOPv2 weights in `data/weights/yolopv2.pt`.
3. Ensure the `utils.utils` module is in the project directory.

## Usage
Run the script with command-line arguments to specify the input folder and other options:

```bash
python SeritTespit.py --source /path/to/image/folder --weights data/weights/yolopv2.pt --img-size 640
```

### Command-Line Arguments
| Argument            | Description                                                                 | Default                |
|---------------------|-----------------------------------------------------------------------------|------------------------|
| `--source`          | Path to the folder where images are added in real-time                      | `data/atakan2`         |
| `--weights`         | Path to YOLOPv2 model weights                                              | `data/weights/yolopv2.pt` |
| `--img-size`        | Inference image size (pixels)                                              | `640`                  |
| `--conf-thres`      | Object detection confidence threshold                                      | `0.3`                  |
| `--iou-thres`       | IOU threshold for non-max suppression                                      | `0.45`                 |
| `--device`          | Device to use (`mps` or `cpu`)                                             | `mps`                  |
| `--save-conf`       | Save confidence scores in text labels                                      | `False`                |
| `--save-txt`        | Save detection results to `.txt` files                                     | `False`                |
| `--nosave`          | Do not save processed images                                               | `False`                |
| `--classes`         | Filter detections by class IDs (e.g., `0 1`)                               | None                   |
| `--agnostic-nms`    | Use class-agnostic non-max suppression                                     | `False`                |
| `--project`         | Directory to save results                                                  | `runs`                 |
| `--name`            | Experiment name for output folder                                          | `exp`                  |
| `--exist-ok`        | Allow overwriting existing output directory                                | `False`                |
| `--no-visualize`    | Disable visualization of lane lines and drivable area                      | Visualization enabled   |
| `--no-include-mesh` | Exclude drivable area mesh points from output                              | Mesh included          |
| `--language`        | Language for display text (`en` or `tr`)                                   | `tr` (Turkish)         |

### Operation
1. The script monitors the specified `--source` folder for new `.jpg` or `.png` images.
2. When a new image is detected, it is processed using the YOLOPv2 model.
3. Results are visualized (if enabled) in two windows:
   - **Processed Video**: Original image with bounding boxes, drivable area, lane lines, and text overlays.
   - **Birdseye View**: BEV image with lane lines, centerlines, drivable area boundary, and mesh points.
4. Processed images and BEV images are saved to the output directory (`runs/exp` or as specified).
5. Detection data (objects, lane lines, drivable area, metrics) is stored in `detections.json`.
6. The script waits for new images if none are available, checking every 0.1 seconds.
7. Press the **Escape** key to stop the script and save final results.

## Outputs
Results are saved in the output directory (`runs/exp` or as specified by `--project` and `--name`):
- **Processed Images**: Original images with bounding boxes and annotations (`<image_name>.jpg`).
- **BEV Images**: Bird's-eye view images with lane lines, centerlines, and drivable area (`bev_<image_name>.jpg`).
- **Text Labels** (if `--save-txt`): Detection results in YOLO format (`labels/<image_name>.txt`).
- **JSON Output**: `detections.json` containing per-frame data:
  ```json
  [
    {
      "frame_id": int,
      "timestamp": float,
      "image_shape": [height, width],
      "objects": [{"bbox": [x1, y1, x2, y2], "class_id": int, "class_name": str, "confidence": float, "center": [x, y]}, ...],
      "lane_lines": [{"points": [[x, y], ...]}, ...],
      "lane_centerlines": [{"points": [[x, y], ...]}, ...],
      "drivable_area": {"boundary_points": [[x, y], ...], "mesh_points": [[x, y], ...], "area": int, "mask_shape": [height, width]},
      "lane_count": int,
      "metrics": {"inference_time": float, "nms_time": float, "total_time": float},
      "errors": [str, ...]
    },
    ...
  ]
  ```

## Notes
- **Performance**: The script checks for new images every 0.1 seconds to balance responsiveness and CPU usage. Adjust the `time.sleep(0.1)` value in `detect()` for different intervals.
- **File Types**: Only `.jpg` and `.png` images are processed. Modify the `glob.glob` patterns in `detect()` to support other formats (e.g., `.jpeg`).
- **Visualization**: Disable visualization with `--no-visualize` to reduce resource usage for headless operation.
- **BEV Dimensions**: The bird's-eye view transformation uses hardcoded parameters (`f=80`, `s=7`, `h=40`). These may need adjustment for accurate spatial mapping.
- **Lane Smoothing**: Lane lines are fitted with a 10th-degree polynomial, which may overfit. Future improvements could use lower-degree polynomials or alternative smoothing methods.
- **Device Support**: Supports MPS (Apple Silicon) and CPU. CUDA is not implemented.
- **Error Handling**: The script skips invalid images and logs errors in the JSON output.

## Limitations
- Rapid image additions may lead to processing delays if inference is slower than file creation. Consider batch processing or a queue system for high-frequency inputs.
- The script assumes images are valid and properly formatted. Corrupted files may cause errors.
- Video input is not supported in this version (focus is on real-time image folder monitoring).

## Customization
- **Add File Types**: Modify the `glob.glob` patterns in `detect()` (e.g., add `*.jpeg`).
- **Adjust Sleep Interval**: Change `time.sleep(0.1)` in `detect()` for faster/slower folder checks.
- **BEV Calibration**: Update `f`, `s`, `h`, and `alf` in `birdeye()` to match your camera's field of view.
- **Smoothing**: Replace the polynomial fitting in `get_lane_points()` with a lower-degree polynomial or spline for better lane line stability.
