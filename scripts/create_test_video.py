"""Create a short test video for the pipeline."""
import cv2
import numpy as np
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUTPUT = BASE / "test_video.mp4"

def main():
    # 5 seconds at 10 fps, 640x480
    fps = 10
    duration_sec = 5
    w, h = 640, 480
    total_frames = fps * duration_sec

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, fps, (w, h))

    for i in range(total_frames):
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)  # Dark gray background
        # Draw a simple "person" rectangle to simulate a classroom
        cv2.rectangle(frame, (200, 150), (250, 350), (100, 100, 100), -1)
        cv2.rectangle(frame, (350, 150), (400, 350), (100, 100, 100), -1)
        cv2.putText(frame, f"Frame {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()
