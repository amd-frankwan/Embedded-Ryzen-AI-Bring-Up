import collections
import sys
import time
import queue
import cv2

from common import (
    create_cpu_session,
    create_npu_session,
    sort_frame_func,
    onnx_model_path,
    preprocess_func,
    infer_func,
    postprocess_func,
    start_threads,
    FPS,
)

# App options
control_fps = True
#control_fps = False
line_width = 2
#video_path = r"C:\Users\juna\OneDrive - Advanced Micro Devices Inc\Embedded-x86\Demo\Ryzen-AI\Videos\video-object-detection-1.mp4"
video_path = 0
play_loop = True
save_video = False

# Start video capture
print("Start video capture", video_path)
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():  
    print("Error: Couldn't open the video file!")  
    sys.exit()

video_fps = cap.get(cv2.CAP_PROP_FPS)
print(f"{video_fps=}")
video_fps = 15

# Record video
video_out = None
if save_video:
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_out = cv2.VideoWriter('output.avi', fourcc, 15.0, (1920,  1080))

# Create ONNX runtime session
#session = create_cpu_session(onnx_model_path)
session = create_npu_session(onnx_model_path)

# Queues and task threads
pre_q     = queue.Queue(maxsize=1)
infer_q   = queue.Queue(maxsize=1)
sort_q    = queue.Queue(maxsize=0) # set maxsize > 0 will cause deadlock when exiting app
post_q    = queue.Queue(maxsize=0)
display_q = queue.Queue(maxsize=0)

start_threads(preprocess_func, pre_q, infer_q, num_threads=1)
start_threads(infer_func, infer_q, sort_q, (session,), num_threads=4)
start_threads(sort_frame_func, sort_q, post_q, num_threads=1)
start_threads(postprocess_func, post_q, display_q, (line_width,), num_threads=1)

# Variables for fps control
video_start_time = None
frame_count = 0

# Variables for fps measuring
fps_counter = FPS()

# Variables for ordering output frames
input_frame_id = 0
buffered_frame = collections.deque(maxlen=5)
out_frame = None

# Main loop
while True:
    if video_start_time is None:
        video_start_time = time.time()
    
    if control_fps:
        # Calculate target time for current frame
        target_time = video_start_time + (frame_count / video_fps)
        current_time = time.time()
        
        # Sleep if we're ahead of schedule
        sleep_duration = target_time - current_time
        if sleep_duration > 0:
            time.sleep(sleep_duration)
    
    for _ in range(2):
        ret, frame = cap.read()
    frame_count += 1

    if not ret:
        if play_loop:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_count = 0
            video_start_time = None
            continue
        else:
            break

    pre_q.put((frame, input_frame_id))
    input_frame_id += 1

    try:
        while True:
            buffered_frame.append(display_q.get(block=False))
            fps_counter() # Increment counter
    except queue.Empty:
        pass

    if len(buffered_frame) > 0:
        out_frame, = buffered_frame.popleft()

        # Draw fps
        label = f"YOLOv8m FPS : {fps_counter.get():.1f}"
        cv2.putText(out_frame, label, (10, 40), 0, 1, (0,  0,0), thickness=3, lineType=cv2.LINE_AA)
        cv2.putText(out_frame, label, (10, 40), 0, 1, (0,255,0), thickness=2, lineType=cv2.LINE_AA)

        if video_out is not None:
            video_out.write(out_frame)

    if out_frame is not None:
        cv2.imshow('Video', out_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("finish app")
        break

cap.release()
if video_out is not None:
    video_out.release()
cv2.destroyAllWindows()
