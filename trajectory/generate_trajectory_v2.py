from ultralytics import YOLO
import cv2
import numpy as np
import os
import imageio
from scipy.interpolate import splprep, splev

def generate_trajectory(input_video_path, output_video_path):

    model = YOLO(
        "notebooks/runs/detect/train2/weights/best.pt"
    )

    cap = cv2.VideoCapture(input_video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30



    trajectory = []
    frames = []

    MAX_TRAIL = 10

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, conf=0.1)[0]

        if results.boxes is not None and len(results.boxes) > 0:

            box = results.boxes.xywh.cpu().numpy()[0]
            
            cx, cy, w, h = box
            
            cx = int(cx)
            cy = int(cy)

        
            if len(trajectory) == 0:

                trajectory.append((cx,cy))

            else:

                prev_x, prev_y = trajectory[-1]

                distance = np.sqrt(
                    (cx-prev_x)**2 +
                    (cy-prev_y)**2
                )

                if distance < 40:
                    trajectory.append((cx,cy))

            box = results.boxes.xyxy.cpu().numpy()[0]

            x1, y1, x2, y2 = box

            cv2.rectangle(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0,255,0),
                2
            )

            if len(trajectory)>=5:

                smooth_points=[]

                WINDOW=7

                for i in range(len(trajectory)):

                    start=max(0,i-WINDOW)

                    median_x = int(
                        np.median(
                            [p[0] for p in trajectory[start:i+1]]
                        )
                    )

                    median_y = int(
                        np.median(
                            [p[1] for p in trajectory[start:i+1]]
                        )
                    )

                    smooth_points.append((median_x, median_y))

                    display_points = smooth_points[-MAX_TRAIL:]

                    if len(display_points) >= 10:
                        split_index = int(
                            0.8 * len(display_points)
                        )

                        pre_points = display_points[:split_index]

                        post_points = display_points[split_index:]

                        for points in [pre_points, post_points]:

                            if len(points) >= 5:
                            
                                x = np.array(
                                    [p[0] for p in points]
                                )
                            
                                y = np.array(
                                    [p[1] for p in points]
                                )
                            
                                coefficients = np.polyfit(
                                    x,
                                    y,
                                    2
                                )
                            
                                polynomial = np.poly1d(
                                    coefficients
                                )
                            
                                x_curve = np.linspace(
                                    x.min(),
                                    x.max(),
                                    300
                                )
                            
                                y_curve = polynomial(
                                    x_curve
                                )
                            
                                curve_points = np.array(
                                    [
                                        [int(xc), int(yc)]
                                        for xc, yc in zip(
                                            x_curve,
                                            y_curve
                                        )
                                    ]
                                )
                            
                                cv2.polylines(
                                    frame,
                                    [curve_points],
                                    False,
                                    (0,0,255),
                                    5
                                )

        frames.append(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        )

    print("Output video path:", output_video_path)
    cap.release()
    imageio.mimsave(
        output_video_path,
        frames,
        fps=fps
    )

    print("Video exists:", os.path.exists(output_video_path))

    if os.path.exists(output_video_path):
        print("Video size:", os.path.getsize(output_video_path))