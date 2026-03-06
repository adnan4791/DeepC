#Extract Y Frame dari Video
import numpy as np
import os

def read_yuv420(filename, width, height, num_frames):

    frame_size = width * height * 3 // 2

    with open(filename, 'rb') as f:
        raw = f.read()

    frames = []

    for i in range(num_frames):

        start = i * frame_size

        y = np.frombuffer(
            raw[start:start + width*height],
            dtype=np.uint8
        ).reshape((height, width))

        frames.append(y)

    return frames


width = 176
height = 144
num_frames = 150

video_path = "suzie_qcif.yuv"

frames = read_yuv420(video_path, width, height, num_frames)

os.makedirs("dataset/y_frames", exist_ok=True)

for i, frame in enumerate(frames):

    np.save(f"dataset/y_frames/frame_{i:03d}.npy", frame)

print("DONE")

# Validasi Extract Y Frame
# import numpy as np
# import matplotlib.pyplot as plt

# img = np.load("dataset/y_frames/frame_000.npy")

# plt.imshow(img, cmap='gray')
# plt.show()