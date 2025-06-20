import argparse
from pathlib import Path
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('qt5agg')
import time

# Conclude setting / general reprocessing / plots / metrices / datasets
from utils.utils import \
    time_synchronized,select_device, increment_path, \
    scale_coords,xyxy2xywh,non_max_suppression,split_for_trace_model, \
    driving_area_mask,lane_line_mask,plot_one_box,show_seg_result, \
    AverageMeter, \
    LoadImages



def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='data/weights/yolopv2.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/part1.mp4', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    return parser

def birdeye(image: np.ndarray,source_points,cm,farcor,nearcor,height) -> np.ndarray:

    """Applies a perspective transformation (bird's eye view) to the given image.

    Parameters:
        image (np.ndarray): The input image to be transformed.

    Returns:
        np.ndarray: The transformed image with a bird's eye view."""

    # Define the source points (manually selected)
    pt_A = source_points[0]
    pt_B = source_points[1]
    pt_C = source_points[2]
    pt_D = source_points[3]

    # Calculate the maximum width and height for the transformed image
    """width_AD = np.sqrt(((pt_A[0] - pt_D[0]) ** 2) + ((pt_A[1] - pt_D[1]) ** 2))
    width_BC = np.sqrt(((pt_B[0] - pt_C[0]) ** 2) + ((pt_B[1] - pt_C[1]) ** 2))
    maxWidth = max(int(width_AD), int(width_BC))

    height_AB = np.sqrt(((pt_A[0] - pt_B[0]) ** 2) + ((pt_A[1] - pt_B[1]) ** 2))
    height_CD = np.sqrt(((pt_C[0] - pt_D[0]) ** 2) + ((pt_C[1] - pt_D[1]) ** 2))
    maxHeight = max(int(height_AB), int(height_CD))"""

    # Define input and output points for the perspective transform
    input_pts = np.float32([pt_A, pt_B, pt_C, pt_D])
    cm = cm

    output_pts = np.float32([[0, 0],
                             [(farcor-nearcor)/2* cm, height * cm],
                             [(farcor+nearcor)/2 * cm, height * cm],
                             [farcor * cm, 0]])

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(input_pts, output_pts)

    # Apply the perspective transformation
    transformed_img = cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

    return transformed_img


# Example usage:
# img = cv2.imread('path_to_image.jpg')
# transformed = birdeye(img)
# cv2.imshow('Transformed Image', transformed)
# cv2.waitKey(0)


def detect():
    # setting and directories
    source, weights, save_txt, imgsz = opt.source, opt.weights, opt.save_txt, opt.img_size
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images

    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    inf_time = AverageMeter()
    waste_time = AverageMeter()
    nms_time = AverageMeter()

    # Load model
    stride = 32
    model = torch.jit.load(weights)
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    model = model.to(device)

    if half:
        model.half()  # to FP16
    model.eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()

    # Create a named window for displaying the video
    cv2.namedWindow('Processed Video', cv2.WINDOW_NORMAL)

    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        [pred, anchor_grid], seg, ll = model(img)
        t2 = time_synchronized()

        # waste time: the incompatibility of torch.jit.trace causes extra time consumption in demo version
        # but this problem will not appear in official version
        tw1 = time_synchronized()
        pred = split_for_trace_model(pred, anchor_grid)
        tw2 = time_synchronized()

        # Apply NMS
        t3 = time_synchronized()
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t4 = time_synchronized()

        da_seg_mask = driving_area_mask(seg)
        ll_seg_mask = lane_line_mask(ll)

        # SURULEBILIR ALAN HESABI
        driving_area_size = np.sum(da_seg_mask == 1)

        # SERIT SAYISI HESABI (FLICKERING)
        ll_seg_mask_uint8 = (ll_seg_mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(ll_seg_mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        lane_count = len(contours)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    # s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:  # Add bbox to image
                        plot_one_box(xyxy, im0, line_thickness=3)

            # Print time (inference)
            # BIRDEYE olayı için GERÇEK ÖLÇÜLERİ GİR ve uncommentle       birdeye(im0,source_points=,cm=,farcor=,nearcor=,height=)
            print(f'{s}Done. ({t2 - t1:.3f}s)')
            show_seg_result(im0, (da_seg_mask, ll_seg_mask), is_demo=True)


            cv2.putText(im0, f'Surulebilir Alan: {driving_area_size} piksel', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(im0, f'Serit Sayisi: {lane_count}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Display the processed frame

            cv2.imshow('Processed Video', im0)
            if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
                break

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w, h = im0.shape[1], im0.shape[0]
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    inf_time.update(t2 - t1, img.size(0))
    nms_time.update(t4 - t3, img.size(0))
    waste_time.update(tw2 - tw1, img.size(0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))
    print(f'Done. ({time.time() - t0:.3f}s)')

    # Release the video writer and close the display window
    if vid_writer is not None:
        vid_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    opt =  make_parser().parse_args()
    print(opt)

    with torch.no_grad():
        detect()
