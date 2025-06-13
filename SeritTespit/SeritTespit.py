# democarlavv.py (Server - ZMQ and no-vis by default)
# REPSOCKET KULLANARAK INPUT ALIYOR
# TERMİNALE HER SUBPROCESSİN SÜRESİNİ YAZIYOR
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
import zmq

# Import utilities
from utils.utils import (
    time_synchronized, increment_path, 
    scale_coords, xyxy2xywh, non_max_suppression, split_for_trace_model,
    driving_area_mask, lane_line_mask, plot_one_box, show_seg_result,
    AverageMeter, LoadImages
)

# --- (No changes to get_lane_points, get_drivable_area_mesh, birdeye, select_device) ---
def get_lane_points(ll_seg_mask, min_area=100):
    lane_lines = []
    ll_seg_mask_uint8 = (ll_seg_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(ll_seg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area: continue
        points = cnt.squeeze()
        if points.ndim == 1: points = points.reshape(1, -1)
        if len(points) >= 2:
            x, y = points[:, 0], points[:, 1]
            try:
                poly_coeffs = np.polyfit(y, x, 10)
                poly_func = np.poly1d(poly_coeffs)
                y_smooth = np.linspace(y.min(), y.max(), num=50)
                x_smooth = poly_func(y_smooth)
                smooth_line = np.column_stack((x_smooth, y_smooth)).astype(int)
                lane_lines.append(smooth_line)
            except np.linalg.LinAlgError: continue
    return lane_lines

def get_drivable_area_mesh(da_seg_mask, step=50, include_mesh=True):
    da_seg_mask_uint8 = (da_seg_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(da_seg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return np.array([]), np.array([])
    main_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.005 * cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, epsilon, True)
    boundary_points = approx.squeeze()
    mesh_points = np.array([])
    if include_mesh:
        x, y = np.arange(0, da_seg_mask.shape[1], step), np.arange(0, da_seg_mask.shape[0], step)
        xx, yy = np.meshgrid(x, y)
        mask = da_seg_mask[yy, xx] > 0
        mesh_points = np.column_stack([xx[mask], yy[mask]])
    return boundary_points, mesh_points

def select_device(device=''):
    if device.lower() == 'mps' and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')

def birdeye(img, source_points):
    f,s,h,alf=5,1,2,1280/5
    output_pts = np.float32([[0, 0],[(f-s)*alf/2, h*alf],[(f+s)*alf/2, h*alf],[f*alf, 0]])
    src_points = np.float32(source_points)
    M = cv2.getPerspectiveTransform(src_points, output_pts)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped, M


def make_parser():
    parser = argparse.ArgumentParser()
    
    # --- MODIFIED: Set ZMQ REP socket as the default mode ---
    parser.add_argument('--repsocket', type=str, default='tcp://*:5555', 
                        help='Run as a ZMQ REP server on this address. Set to "" to disable and run in file mode.')
    
    # --- MODIFIED: Make visualization off by default. Use a flag to turn it ON. ---
    parser.add_argument('--visualize', action='store_true', 
                        help='Enable visualization of perception results (off by default).')

    # --- Other arguments remain the same ---
    parser.add_argument('--weights', type=str, default='data/weights/yolopv2.pt', help='model.pt path')
    parser.add_argument('--source', type=str, default='data/new_map', help='source (used only if --repsocket is disabled)')
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
    parser.add_argument('--no-include-mesh', action='store_false', default=True, dest='include_mesh', help='exclude drivable area mesh points from output')
    parser.add_argument('--language', type=str, default='tr', choices=['en', 'tr'], help='language for display text')
    return parser


# --- (No other changes are needed in the rest of the file) ---
# --- (The process_frame, detect, run_file_based_processing functions are identical) ---
def process_frame(im0s, model, device, half, opt, frame_id=0):
    imgsz = opt.img_size
    CLASS_NAMES = {0: "car", 1: "pedestrian", 2: "truck"}
    x_resize, y_resize = 2, 9/8
    source_points = [[21*x_resize, 361*y_resize], [21*x_resize, 561*y_resize], [620*x_resize, 561*y_resize], [620*x_resize, 361*y_resize]]
    bev_height = 800

    # Padded resize
    img = cv2.resize(im0s, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    img = img.transpose(2, 0, 1) # HWC to CHW
    img = np.ascontiguousarray(img)

    t_frame_start = time_synchronized()
    
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()
    img /= 255.0
    
    output_frame = {
        "frame_id": frame_id, "timestamp": time.time(), "image_shape": [bev_height, im0s.shape[1]],
        "objects": [], "lane_lines": [], "lane_centerlines": [], "drivable_area": {}, "metrics": {}, "errors": []
    }
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    
    t_preproc_end = time_synchronized()

    try:
        [pred, anchor_grid], seg, ll = model(img)
    except RuntimeError as e:
        output_frame["errors"].append(f"Inference error: {str(e)}")
        return output_frame
    
    t_inference_end = time_synchronized()

    pred = split_for_trace_model(pred, anchor_grid)
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    t_nms_end = time_synchronized()
    
    da_seg_mask = driving_area_mask(seg)
    ll_seg_mask = lane_line_mask(ll)
    t_mask_creation_end = time_synchronized()

    _, M = birdeye(im0s, source_points)
    da_seg_mask_uint8 = (da_seg_mask * 255).astype(np.uint8)
    ll_seg_mask_uint8 = (ll_seg_mask * 255).astype(np.uint8)
    bev_da_seg_mask = cv2.warpPerspective(da_seg_mask_uint8, M, (im0s.shape[1], bev_height), flags=cv2.INTER_NEAREST)
    bev_ll_seg_mask = cv2.warpPerspective(ll_seg_mask_uint8, M, (im0s.shape[1], bev_height), flags=cv2.INTER_NEAREST)
    bev_da_seg_mask = (bev_da_seg_mask > 127).astype(np.uint8)
    bev_ll_seg_mask = (bev_ll_seg_mask > 127).astype(np.uint8)
    t_bev_end = time_synchronized()

    lane_lines = get_lane_points(bev_ll_seg_mask)
    for line in lane_lines: output_frame["lane_lines"].append({"points": line.tolist()})
    lane_count = len(lane_lines)
    output_frame["lane_count"] = lane_count
    t_lane_extract_end = time_synchronized()

    lane_centerlines = []
    if len(lane_lines) > 1:
        for i in range(len(lane_lines) - 1):
            line1, line2 = np.array(lane_lines[i]), np.array(lane_lines[i + 1])
            min_length = min(len(line1), len(line2))
            centerline = ((line1[:min_length] + line2[:min_length]) / 2).astype(int)
            lane_centerlines.append(centerline)
    for centerline in lane_centerlines: output_frame["lane_centerlines"].append({"points": centerline.tolist()})
    t_centerline_end = time_synchronized()

    da_boundary, da_mesh = get_drivable_area_mesh(bev_da_seg_mask, step=50, include_mesh=opt.include_mesh)
    output_frame["drivable_area"] = {
        "boundary_points": da_boundary.tolist() if da_boundary.size > 0 else [],
        "mesh_points": da_mesh.tolist() if da_mesh.size > 0 else [],
        "mask_shape": [bev_height, im0s.shape[1]]
    }
    t_drivable_extract_end = time_synchronized()

    im0_vis = im0s.copy()
    for i, det in enumerate(pred):
        if len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            for *xyxy, conf, cls in reversed(det):
                class_id = int(cls.item())
                output_frame["objects"].append({
                    "bbox": [int(x.item()) for x in xyxy], "class_id": class_id,
                    "class_name": CLASS_NAMES.get(class_id, "unknown"), "confidence": float(conf.item()),
                    "center": [float((xyxy[0] + xyxy[2]) / 2), float((xyxy[1] + xyxy[3]) / 2)]
                })
                if opt.visualize: plot_one_box(xyxy, im0_vis, line_thickness=3)
    t_object_proc_end = time_synchronized()
    
    if opt.visualize:
        warped_img_vis, _ = birdeye(im0s, source_points)
        for lane in output_frame["lane_lines"]:
            points = np.array(lane["points"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(warped_img_vis, [points], isClosed=False, color=(0, 0, 255), thickness=2)
        if output_frame["drivable_area"]["boundary_points"]:
            boundary = np.array(output_frame["drivable_area"]["boundary_points"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(warped_img_vis, [boundary], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.imshow('Birdseye View', warped_img_vis)
        
        show_seg_result(im0_vis, (da_seg_mask, ll_seg_mask), is_demo=True)
        lane_text = f"Serit Sayisi: {lane_count}" if opt.language == 'tr' else f"Lane Count: {lane_count}"
        cv2.putText(im0_vis, lane_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.imshow('Processed Video', im0_vis)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            raise KeyboardInterrupt 

    t_vis_end = time_synchronized()

    timing_data = {
        "1_preprocessing_ms": (t_preproc_end - t_frame_start) * 1000, "2_inference_ms": (t_inference_end - t_preproc_end) * 1000,
        "3_obj_det_nms_ms": (t_nms_end - t_inference_end) * 1000, "4_seg_mask_creation_ms": (t_mask_creation_end - t_nms_end) * 1000,
        "5_bev_transform_ms": (t_bev_end - t_mask_creation_end) * 1000, "6_lane_extraction_ms": (t_lane_extract_end - t_bev_end) * 1000,
        "7_centerline_calc_ms": (t_centerline_end - t_lane_extract_end) * 1000, "8_drivable_area_extraction_ms": (t_drivable_extract_end - t_centerline_end) * 1000,
        "9_object_processing_ms": (t_object_proc_end - t_drivable_extract_end) * 1000, "10_visualization_ms": (t_vis_end - t_object_proc_end) * 1000,
        "total_frame_time_ms": (t_vis_end - t_frame_start) * 1000
    }
    output_frame["metrics"]["timing_details_ms"] = timing_data
    return output_frame

def detect():
    opt = make_parser().parse_args()
    print(opt)

    device = select_device(opt.device)
    try:
        model = torch.jit.load(opt.weights, map_location=device)
        model = model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    half = device.type != 'cpu'
    if half:
        model.half()
    model.eval()

    if device.type != 'cpu':
        model(torch.zeros(1, 3, opt.img_size, opt.img_size).to(device).type_as(next(model.parameters())))

    if opt.repsocket:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.bind(opt.repsocket)
        print(f"ZMQ REP server listening on {opt.repsocket}")

        if opt.visualize:
            cv2.namedWindow('Birdseye View', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)
        
        while True:
            try:
                image_bytes = socket.recv()
                nparr = np.frombuffer(image_bytes, np.uint8)
                im0s = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if im0s is None:
                    raise ValueError("Failed to decode image from bytes.")
                
                output_frame = process_frame(im0s, model, device, half, opt)
                
                reply_json = json.dumps(output_frame, cls=NumpyEncoder)
                socket.send_string(reply_json)
            
            except KeyboardInterrupt:
                print("\nCaught KeyboardInterrupt, shutting down server.")
                break
            except Exception as e:
                print(f"An error occurred: {e}")
                error_reply = json.dumps({"error": str(e)})
                socket.send_string(error_reply)
        
        cv2.destroyAllWindows()
        socket.close()
        context.term()

    else:
        run_file_based_processing(opt, model, device, half)


def run_file_based_processing(opt, model, device, half):
    save_img = not opt.nosave and not opt.source.endswith('.txt')
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))
    (save_dir / 'labels' if opt.save_txt else save_dir).mkdir(parents=True, exist_ok=True)
    
    dataset = LoadImages(opt.source, img_size=opt.img_size, stride=32)
    vid_path, vid_writer = None, None
    output_list = []

    timing_keys = ["1_preprocessing_ms", "2_inference_ms", "3_obj_det_nms_ms", "4_seg_mask_creation_ms", "5_bev_transform_ms", "6_lane_extraction_ms", "7_centerline_calc_ms", "8_drivable_area_extraction_ms", "9_object_processing_ms", "10_visualization_ms", "total_frame_time_ms"]
    avg_timing_accumulator = {key: 0.0 for key in timing_keys}
    frame_count = 0
    t0 = time.time()

    try:
        if opt.visualize:
            cv2.namedWindow('Birdseye View', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)

        for path, _, im0s, vid_cap in dataset:
            output_frame = process_frame(im0s, model, device, half, opt, frame_id=dataset.frame)
            output_list.append(output_frame)
            
            if "timing_details_ms" in output_frame["metrics"]:
                for key, value in output_frame["metrics"]["timing_details_ms"].items():
                    avg_timing_accumulator[key] += value
                frame_count += 1
            
            print(f"Frame {output_frame['frame_id']} done. Total time: {output_frame['metrics'].get('timing_details_ms', {}).get('total_frame_time_ms', 0):.2f}ms")

    except KeyboardInterrupt:
        print("Processing stopped by user.")
    finally:
        if vid_writer is not None: vid_writer.release()
        cv2.destroyAllWindows()

    print("\n" + "="*50)
    print(" " * 10 + "Average Performance Metrics")
    print("="*50)
    if frame_count > 0:
        avg_timings = {key: value / frame_count for key, value in avg_timing_accumulator.items()}
        max_key_len = max(len(key) for key in avg_timings.keys())
        for key, avg_value in avg_timings.items():
            print(f"{key:<{max_key_len}} : {avg_value:>8.2f} ms/frame")
        avg_fps = 1000.0 / avg_timings['total_frame_time_ms']
        print("-" * 50)
        print(f"{'Average FPS':<{max_key_len}} : {avg_fps:>8.2f}")
    else:
        print("No frames were processed.")
    print("="*50)
    print(f'\nTotal process wall-clock time: {time.time() - t0:.3f}s')

    with open(f"{save_dir}/detections.json", "w") as f:
        json.dump(output_list, f, indent=2, cls=NumpyEncoder)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.floating, torch.Tensor)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

if __name__ == '__main__':
    with torch.no_grad():
        detect()
