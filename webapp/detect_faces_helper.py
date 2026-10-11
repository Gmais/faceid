#!/usr/bin/env python3
"""
Subprocess helper: detects faces in a video frame or image using OpenCV Haar cascade.
Outputs JSON with face crops as base64 thumbnails and positional info.
Faces are sorted left-to-right to match FaceFusion's default ordering.

Usage: python detect_faces_helper.py <path> [frame_number]
"""
import sys
import json
import os
import base64

import cv2


def detect_faces_haar(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    cascade = cv2.CascadeClassifier(cascade_path)
    dets = cascade.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(40, 40))
    if not hasattr(dets, '__len__') or len(dets) == 0:
        return []
    # Sort left-to-right (matches FaceFusion's default "left-right" face order)
    return sorted([(x, y, x + w, y + h) for (x, y, w, h) in dets], key=lambda b: b[0])


def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No path given", "faces": []}))
        sys.exit(1)

    path = sys.argv[1]
    frame_number = int(sys.argv[2]) if len(sys.argv) > 2 else -1

    ext = os.path.splitext(path)[1].lower()
    is_video = ext in {'.mp4', '.mov', '.avi', '.webm', '.mkv'}

    if is_video:
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        if frame_number < 0:
            frame_number = max(0, int(total_frames * 0.1))
        frame_number = min(frame_number, total_frames - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            print(json.dumps({
                "error": "Cannot read frame",
                "faces": [],
                "frame_number": frame_number,
                "total_frames": total_frames,
                "fps": fps,
            }))
            sys.exit(0)
    else:
        frame = cv2.imread(path)
        if frame is None:
            print(json.dumps({"error": "Cannot read image", "faces": []}))
            sys.exit(0)
        total_frames = 1
        fps = 25.0
        frame_number = 0

    boxes = detect_faces_haar(frame)
    result = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        pad = int((x2 - x1 + y2 - y1) * 0.15)
        cx1 = max(0, x1 - pad)
        cy1 = max(0, y1 - pad)
        cx2 = min(frame.shape[1], x2 + pad)
        cy2 = min(frame.shape[0], y2 + pad)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            continue
        thumb = cv2.resize(crop, (120, 120))
        _, buf = cv2.imencode('.jpg', thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = base64.b64encode(buf).decode()
        result.append({
            "position": i,
            "frame_number": frame_number,
            "thumbnail": f"data:image/jpeg;base64,{b64}",
        })

    print(json.dumps({
        "faces": result,
        "frame_number": frame_number,
        "total_frames": total_frames,
        "fps": fps,
    }))


if __name__ == '__main__':
    main()
