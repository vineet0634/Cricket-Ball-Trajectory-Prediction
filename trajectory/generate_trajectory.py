from ultralytics import YOLO
import cv2
import numpy as np

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

    out = cv2.VideoWriter(
        output_video_path,
        cv2.VideoWriter_fourcc(*'XVID'),
        fps,
        (width, height)
    )

    trajectory = []

    MAX_TRAIL = 60

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        results = model(frame, conf=0.1)[0]

        if results.boxes is not None and len(results.boxes) > 0:

            box = results.boxes.xyxy.cpu().numpy()[0]

            x1, y1, x2, y2 = box

            cx = int((x1+x2)/2)
            cy = int((y1+y2)/2)

        
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

                    if len(display_points) >= 2:

                        cv2.polylines(
                            frame,
                            [np.array(display_points)],
                            False,
                            (0,0,255),
                            3
                        )

        out.write(frame)

    print("Output video path:", output_video_path)
    cap.release()
    out.release()
    import os

    print("Video exists:", os.path.exists(output_video_path))

    if os.path.exists(output_video_path):
        print("Video size:", os.path.getsize(output_video_path))
    