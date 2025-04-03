import cv2
import torch
import numpy as np
import threading
import queue
import os
from time import time, sleep
from torchvision.transforms.v2 import CenterCrop, Normalize
from torchvision.transforms import Compose, Lambda
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale
from skimage.metrics import structural_similarity as ssim
from model import Model  


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

feat_model_name = 'x3d_l'
feat_model = torch.hub.load('facebookresearch/pytorchvideo', feat_model_name, pretrained=True)
feat_model = feat_model.eval()
feat_model = feat_model.to(DEVICE)
del feat_model.blocks[-1]  

mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
model_transform_params = {
    "x3d_xs": {"side_size": 182, 
        "crop_size": 182, 
        "num_frames": 4, 
        "sampling_rate": 12},
    "x3d_s": {"side_size": 182, 
        "crop_size": 182, 
        "num_frames": 13, 
        "sampling_rate": 6},
    "x3d_m": {"side_size": 256, 
        "crop_size": 256, 
        "num_frames": 16, 
        "sampling_rate": 5},
    "x3d_l": {"side_size": 320, 
        "crop_size": 320, 
        "num_frames": 16, 
        "sampling_rate": 5},
}


transform_params = model_transform_params[feat_model_name]
transform = ApplyTransformToKey(
    key="video",
    transform=Compose(
        [
            Lambda(lambda x: x / 255.0),
            Normalize(mean, std),
            ShortSideScale(size=transform_params["side_size"]),
            CenterCrop((transform_params["crop_size"], transform_params["crop_size"])),
            Lambda(lambda x: x.permute((1, 0, 2, 3)))
        ]
    ),
)


CHECKPOINT_PATH = "saved_models/888tiny.pkl"  
THRESHOLD = 0.9
FRAME_COUNT = 16  
SIMILARITY_THRESHOLD = 0.7

process_queue = queue.Queue(maxsize=FRAME_COUNT * 2)
latest_result = None
result_lock = threading.Lock()
pause_event = threading.Event()

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(checkpoint, strict=False)  # load state_dict
    print("Load successfully!")

def draw_text_overlay(frame, text, org, font=cv2.FONT_HERSHEY_SIMPLEX, 
                      font_scale=2.0, text_color=(255, 255, 255), 
                      bg_color=(0, 0, 0), thickness=3, padding=5, alpha=0.6):
    overlay = frame.copy()
    (w, h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = org
    cv2.rectangle(overlay, (x - padding, y - h - padding), (x + w + padding, y + padding), bg_color, -1)
    cv2.putText(overlay, text, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

def video_to_tensor(frames):
    # Change list frame become tensor with shape (num_frames, 3, H, W)
    frames = np.stack(frames, axis=0)  # Shape: (num_frames, H, W, 3)
    tensor = torch.from_numpy(frames).permute(0, 3, 1, 2).float()  # Shape: (num_frames, 3, H, W)
    return tensor

def read_and_display_video(video_path):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    cv2.namedWindow("Anomaly Detection", cv2.WINDOW_NORMAL)
    c = False
    while cap.isOpened():
        if pause_event.is_set():
            key = cv2.waitKey(30) & 0xFF
            if key == ord('p'):
                pause_event.clear()
                print("Resume")
            elif key == ord('q'):
                break
            continue

        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            score = ssim(gray_prev, gray_current)
            if score >= SIMILARITY_THRESHOLD:
                try:
                    process_queue.put_nowait(frame)
                    prev_frame = frame
                except queue.Full:
                    pass
        else:
            try:
                process_queue.put_nowait(frame)
                prev_frame = frame
            except queue.Full:
                pass

        with result_lock:
            current_result = latest_result

        if current_result is not None:
            if current_result > THRESHOLD:
                color = (0, 0, 255)
                status = "Pham phap"
            else:
                color = (0, 255, 0)
                status = "Binh thuong"
            anomaly_text = f"Prob: {current_result:.4f} | {status}"
            cv2.rectangle(frame, (0, 0), (frame.shape[1]-2, frame.shape[0]-2), color, 2)
            draw_text_overlay(frame, anomaly_text, (10, 20), font_scale=0.4, thickness=1, alpha=0.6, text_color=color)

        cv2.imshow("Anomaly Detection", frame)
        if not c:
            cv2.resizeWindow("Anomaly Detection", 1024, 768)
            c = True
        key = cv2.waitKey(30) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            pause_event.set()
            print("Paused")

    cap.release()
    process_queue.put(None)
    cv2.destroyAllWindows()

def process_inference(feat_model, model):
    batch_frames = []
    while True:
        if pause_event.is_set():
            sleep(0.1)
            continue

        frame = process_queue.get()
        if frame is None:
            break

        batch_frames.append(frame)
        if len(batch_frames) >= FRAME_COUNT:
            tensor = video_to_tensor(batch_frames).to(DEVICE)
            t_transform = transform({'video': tensor})['video']
            with torch.no_grad():
                # t1 = time()          
                features = feat_model(t_transform.unsqueeze(0))              
                logits, _ = model(features)
                output = torch.sigmoid(logits).item()  
                # print("prob", output)
                # print("Processing time:", time() - t1)

            with result_lock:
                global latest_result
                latest_result = output

            batch_frames.clear()

if __name__ == '__main__':
    
    model = Model().to(DEVICE)  
    if os.path.exists(CHECKPOINT_PATH):
        print("Load checkpoint")
        load_checkpoint(model, CHECKPOINT_PATH)
    else:
        print("Khong load checkpoint duoc")
    model.eval()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    video_path = '../../tmp/UCFCrimeDataset/Test/Fighting/Fighting018_x264.mp4'

    thread_inference = threading.Thread(target=process_inference, args=(feat_model, model))
    thread_inference.start()
    read_and_display_video(video_path)
    thread_inference.join()
    
    