import numpy as np
import os

def read_y_frame(filename, width, height, frame_index):

    frame_size = width * height * 3 // 2

    with open(filename, "rb") as f:

        f.seek(frame_index * frame_size)

        y = np.frombuffer(
            f.read(width * height),
            dtype=np.uint8
        ).reshape((height, width))

    return y


width = 352
height = 288
num_frames = 150
runs = 5

os.makedirs("dataset/race", exist_ok=True)

for run in range(runs):

    print("RUN", run+1)

    # jalankan FSRCNN race
    os.system("./fsrcnn_race suzie_qcif.yuv output_race.yuv")

    for frame in range(num_frames):

        img = read_y_frame(
            "output_race.yuv",
            width,
            height,
            frame
        )

        np.save(
            f"dataset/race/frame_{frame:03d}_run{run+1}.npy",
            img
        )

print("DONE")

# VALIDASI
# import numpy as np

# a = np.load("dataset/race/frame_050_run1.npy")
# b = np.load("dataset/race/frame_050_run2.npy")

# print(np.sum(a != b))