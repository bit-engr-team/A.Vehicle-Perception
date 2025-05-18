# SOURCE ARGÜMANIYLA ALDIĞI FOLDERA EKLENEN YENİ FOTOĞRAFLARI İŞLİYOR
# KABUL ETTİĞİ GÖRÜNTÜNÜN DIMENSIONINA GÖRE OLAN AYAR SIKINTILI!!! DÜZELTİLECEK
# ŞUANLIK ESCAPE BASINCA FOLDER'I İZLEMEYİ BIRAKIYOR
import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import time
import os
import json
import glob

# Import utilities
from utils.utils import (
    time_synchronized, increment_path, 
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result,
    AverageMeter
)

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='data/weights/yolopv2.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='data/atakan2', help='source folder for images')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='mps', help='device to use (mps, cpu)')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok')
    parser.add_argument('--no-visualize', action='store_false', default=True, dest='visualize', help='disable visualization of lane and drivable area contours')
    parser.add_argument('--no-include-mesh', action='store_false', default=True, dest='include_mesh', help='exclude drivable area mesh points from output')
    parser.add_argument('--language', type=str, default='tr', choices=['en', 'tr'], help='language for display text')
    return parser

def get_lane_points(ll_seg_mask, min_area=100):
    lane_lines = []
    ll_seg_mask_uint8 = (ll_seg_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(ll_seg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        
        points = cnt.squeeze()
        if points.ndim == 1:
            points = points.reshape(1, -1)

        if len(points) >= 2:
            x = points[:, 0]
            y = points[:, 1]
            try:
                poly_coeffs = np.polyfit(y, x, 10)
                poly_func = np.poly1d(poly_coeffs)
                y_smooth = np.linspace(y.min(), y.max(), num=50)
                x_smooth = poly_func(y_smooth)
                smooth_line = np.column_stack((x_smooth, y_smooth)).astype(int)
                lane_lines.append(smooth_line)
            except np.linalg.LinAlgError:
                continue

    return lane_lines

def get_drivable_area_mesh(da_seg_mask, step=50, include_mesh=True):
    da_seg_mask_uint8 = (da_seg_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(da_seg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return np.array([]), np.array([])
    
    main_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    boundary_points = approx.squeeze()
    
    mesh_points = np.array([])
    if include_mesh:
        x = np.arange(0, da_seg_mask.shape[1], step)
        y = np.arange(0, da_seg_mask.shape[0], step)
        xx, yy = np.meshgrid(x, y)
        mask = da_seg_mask[yy, xx] > 0
        mesh_points = np.column_stack([xx[mask], yy[mask]])
    
    return boundary_points, mesh_points

def select_device(device=''):
    if device.lower() == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def birdeye(img, source_points):
    f=80
    s=7
    h=40
    alf=1280/f
    
    output_pts = np.float32([[0, 0],
                            [(f-s)*alf/2, h*alf],
                            [(f+s)*alf/2, h*alf],
                            [f*alf, 0]])
    src_points = np.float32(source_points)
    M = cv2.getPerspectiveTransform(src_points, output_pts)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped, M

def load_image(img_path, img_size=640, stride=32):
    """Load and preprocess a single image."""
    img0 = cv2.imread(img_path)
    if img0 is None:
        return None, None, None
    img = cv2.resize(img0, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
    img = np.ascontiguousarray(img)
    return img_path, img, img0

def detect():
    source = Path(opt.source)
    weights = opt.weights
    save_txt = opt.save_txt
    imgsz = opt.img_size
    save_img = not opt.nosave

    # Create output directory
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)

    # Initialize timers
    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    device = select_device(opt.device)
    try:
        model = torch.jit.load(weights, map_location=device)
        model = model.to(device)
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("Trying fallback CPU loading...")
        model = torch.jit.load(weights, map_location='cpu')
        model = model.to('cpu')
        device = torch.device('cpu')

    half = device.type != 'cpu'
    if half:
        model.half()
    model.eval()

    # Warmup
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))

    t0 = time.time()
    output_list = []
    processed_files = set()

    CLASS_NAMES = {
        0: "car",
        1: "pedestrian",
        2: "truck",
    }

    source_points = [
        [3, 385],
        [3, 716],
        [1260, 716],
        [1260, 385]
    ]
    bev_height = 800

    if opt.visualize:
        cv2.namedWindow('Birdseye View', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)

    try:
        while True:
            # Get list of image files
            image_files = sorted(glob.glob(str(source / '*.jpg')) + glob.glob(str(source / '*.png')))
            new_files = [f for f in image_files if f not in processed_files]

            if new_files:
                for img_path in new_files:
                    path, img, im0 = load_image(img_path, img_size=imgsz)
                    if img is None:
                        print(f"Failed to load image: {img_path}")
                        processed_files.add(img_path)
                        continue

                    img = torch.from_numpy(img).to(device)
                    img = img.half() if half else img.float()
                    img /= 255.0
                    output_frame = {
                        "frame_id": len(processed_files),
                        "timestamp": time.time(),
                        "image_shape": [bev_height, im0.shape[1]],
                        "objects": [],
                        "lane_lines": [],
                        "lane_centerlines": [],
                        "drivable_area": {},
                        "metrics": {},
                        "errors": []
                    }
                    if img.ndimension() == 3:
                        img = img.unsqueeze(0)

                    # Inference
                    t1 = time_synchronized()
                    try:
                        [pred, anchor_grid], seg, ll = model(img)
                    except RuntimeError as e:
                        output_frame["errors"].append(f"Inference error: {str(e)}")
                        output_list.append(output_frame)
                        processed_files.add(img_path)
                        continue
                    t2 = time_synchronized()

                    # Post-processing
                    tw1 = time_synchronized()
                    pred = split_for_trace_model(pred, anchor_grid)
                    tw2 = time_synchronized()

                    # NMS
                    t3 = time_synchronized()
                    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                    t4 = time_synchronized()

                    # Process masks
                    da_seg_mask = driving_area_mask(seg)
                    ll_seg_mask = lane_line_mask(ll)

                    # Apply bird's-eye view transformation
                    _, M = birdeye(im0, source_points)
                    da_seg_mask_uint8 = (da_seg_mask * 255).astype(np.uint8)
                    ll_seg_mask_uint8 = (ll_seg_mask * 255).astype(np.uint8)
                    bev_da_seg_mask = cv2.warpPerspective(da_seg_mask_uint8, M, (im0.shape[1], bev_height), flags=cv2.INTER_NEAREST)
                    bev_ll_seg_mask = cv2.warpPerspective(ll_seg_mask_uint8, M, (im0.shape[1], bev_height), flags=cv2.INTER_NEAREST)
                    bev_da_seg_mask = (bev_da_seg_mask > 127).astype(np.uint8)
                    bev_ll_seg_mask = (bev_ll_seg_mask > 127).astype(np.uint8)
                    driving_area_size = np.sum(bev_da_seg_mask == 1)

                    # Lane lines
                    lane_lines = get_lane_points(bev_ll_seg_mask)
                    for line in lane_lines:
                        output_frame["lane_lines"].append({
                            "points": line.tolist()
                        })
                    lane_count = len(lane_lines)
                    output_frame["lane_count"] = lane_count

                    # Lane centerlines
                    lane_centerlines = []
                    for i in range(len(lane_lines) - 1):
                        line1 = np.array(lane_lines[i])
                        line2 = np.array(lane_lines[i + 1])
                        min_length = min(len(line1), len(line2))
                        line1 = line1[:min_length]
                        line2 = line2[:min_length]
                        centerline = ((line1 + line2) / 2).astype(int)
                        lane_centerlines.append(centerline)

                    for centerline in lane_centerlines:
                        output_frame["lane_centerlines"].append({
                            "points": centerline.tolist()
                        })

                    # Drivable area
                    da_boundary, da_mesh = get_drivable_area_mesh(bev_da_seg_mask, step=50, include_mesh=opt.include_mesh)
                    output_frame["drivable_area"] = {
                        "boundary_points": da_boundary.tolist() if da_boundary.size > 0 else [],
                        "mesh_points": da_mesh.tolist() if da_mesh.size > 0 else [],
                        "area": driving_area_size,
                        "mask_shape": [bev_height, im0.shape[1]]
                    }

                    # Process detections
                    for i, det in enumerate(pred):
                        p = Path(img_path)
                        save_path = str(save_dir / p.name)
                        txt_path = str(save_dir / 'labels' / p.stem)
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]

                        if len(det):
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()

                            for *xyxy, conf, cls in reversed(det):
                                xyxy = [x.item() if isinstance(x, torch.Tensor) else x for x in xyxy]
                                conf = conf.item() if isinstance(conf, torch.Tensor) else float(conf)
                                cls = cls.item() if isinstance(cls, torch.Tensor) else int(cls)
                                class_id = cls

                                if save_txt:
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
                                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                output_frame["objects"].append({
                                    "bbox": [int(x) for x in xyxy],
                                    "class_id": class_id,
                                    "class_name": CLASS_NAMES.get(class_id, "unknown"),
                                    "confidence": float(conf),
                                    "center": [float((xyxy[0] + xyxy[2]) / 2), float((xyxy[1] + xyxy[3]) / 2)]
                                })
                                if save_img:
                                    plot_one_box(xyxy, im0, line_thickness=3)

                    if not output_frame["objects"]:
                        output_frame["errors"].append("No objects detected")
                    if not output_frame["lane_lines"]:
                        output_frame["errors"].append("No lane lines detected")
                    if not output_frame["drivable_area"]["boundary_points"]:
                        output_frame["errors"].append("No drivable area detected")

                    # Visualize
                    warped_img, _ = birdeye(im0, source_points)
                    if opt.visualize:
                        warped_img_vis = warped_img.copy()
                        for lane in output_frame["lane_lines"]:
                            points = np.array(lane["points"], dtype=np.int32).reshape(-1, 1, 2)
                            cv2.polylines(warped_img_vis, [points], isClosed=False, color=(0, 0, 255), thickness=2)
                        if output_frame["drivable_area"]["boundary_points"]:
                            boundary = np.array(output_frame["drivable_area"]["boundary_points"], dtype=np.int32).reshape(-1, 1, 2)
                            cv2.polylines(warped_img_vis, [boundary], isClosed=True, color=(0, 255, 0), thickness=2)
                        for point in output_frame["drivable_area"]["mesh_points"]:
                            cv2.circle(warped_img_vis, tuple(np.array(point, dtype=np.int32)), 2, (255, 0, 0), -1)

                        for centerline in output_frame["lane_centerlines"]:
                            points = np.array(centerline["points"], dtype=np.int32).reshape(-1, 1, 2)
                            cv2.polylines(warped_img_vis, [points], isClosed=False, color=(255, 228, 181), thickness=2)

                        cv2.imshow('Birdseye View', warped_img_vis)
                        show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)
                        language = opt.language
                        if language == 'en':
                            area_text = f"Drivable Area: {driving_area_size} pixels"
                            lane_text = f"Lane Count: {lane_count}"
                        else:
                            area_text = f"Surulebilir Alan: {driving_area_size} piksel"
                            lane_text = f"Serit Sayisi: {lane_count}"
                        cv2.putText(im0, area_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(im0, lane_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                        cv2.imshow('Processed Video', im0)

                    # Save results
                    if save_img:
                        cv2.imwrite(save_path, im0)
                        cv2.imwrite(str(save_dir / f"bev_{p.name}"), warped_img_vis)
                        print(f"Result saved to: {save_path}")

                    # Update metrics
                    t_end = time_synchronized()
                    output_frame["metrics"] = {
                        "inference_time": t2 - t1,
                        "nms_time": t4 - t3,
                        "total_time": t_end - t1
                    }
                    output_list.append(output_frame)
                    processed_files.add(img_path)
                    print(f'Processed {img_path} ({t2 - t1:.3f}s)')

            # Check for Escape key
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # Escape key
                break

            # Wait for new images if none are available
            if not new_files:
                print(f"No new images in {source}, waiting...")
                time.sleep(0.1)  # Short sleep to avoid high CPU usage

    finally:
        cv2.destroyAllWindows()

    # Print timings
    inf_time.update(t2 - t1, len(processed_files))
    nms_time.update(t4 - t3, len(processed_files))
    waste_time.update(tw2 - tw1, len(processed_files))
    print(f'Inference: {inf_time.avg:.4f}s/frame  NMS: {nms_time.avg:.4f}s/frame')
    print(f'Total time: {time.time() - t0:.3f}s')

    # Save JSON
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, torch.Tensor):
                return obj.tolist() if obj.numel() > 1 else obj.item()
            return super().default(obj)

    with open(f"{save_dir}/detections.json", "w") as f:
        json.dump(output_list, f, indent=2, cls=NumpyEncoder)

    return output_list

if __name__ == '__main__':
    opt = make_parser().parse_args()
    print(opt)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    
    with torch.no_grad():
        detect()
