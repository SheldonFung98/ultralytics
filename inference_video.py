import cv2
import torch
import argparse
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Inference video with YOLOv8 pose model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLOv8 model.")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file.")
    parser.add_argument("--output_path", type=str, default="output.avi", help="Path for the output video.")
    return parser.parse_args()

def load_model(model_path):
    model = YOLO(model_path) 
    return model

def process_video(model, video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        # results = model(frame, imgsz=(160, 256))
        # w, h = frame.shape[1], frame.shape[0]
        # print(w, h)
        # exit(0)
        res = results[0]
        # print(res.boxes)
        # print(res.keypoints)
        # exit(0)

        if res.boxes.xyxy.shape[0]:
            for i in range(res.boxes.conf.shape[0]):
                if res.boxes.conf[i] > 0.7:
                    box = res.boxes.xyxy[i]
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

                    keypoint = res.keypoints.xy[i]
                    for point in keypoint:
                        cv2.circle(frame, (int(point[0]), int(point[1])), 3, (0, 0, 255), -1)

            # if res.boxes.conf > 0.7:

            #     for box in res.boxes.xyxy:
            #         cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)

            #     for keypoint in res.keypoints.xy:
            #         for point in keypoint:
            #             cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)

        out.write(frame)

    cap.release()
    out.release()
    print("Inference completed. Output saved to:", output_path)

if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model_path)
    process_video(model, args.video_path, args.output_path)