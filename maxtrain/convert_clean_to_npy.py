import numpy as np
import os

def read_yuv420_frames(filename, width, height, num_frames):

    frames = []

    with open(filename, "rb") as f:

        for i in range(num_frames):

            # Read Y channel
            y = np.frombuffer(
                f.read(width * height),
                dtype=np.uint8
            ).reshape((height, width))

            # Skip U and V
            f.read(width * height // 2)

            frames.append(y)

    return frames


# =========================
# Parameter
# =========================

width = 352
height = 288
num_frames = 150

input_file = "suzie_qcif_serial_hr.yuv"

output_dir = "dataset/clean"

os.makedirs(output_dir, exist_ok=True)

# =========================
# Read frames
# =========================

frames = read_yuv420_frames(
    input_file,
    width,
    height,
    num_frames
)

# =========================
# Save as numpy
# =========================

for i, frame in enumerate(frames):

    np.save(
        f"{output_dir}/frame_{i:03d}.npy",
        frame
    )

print("DONE: Saved", len(frames), "frames")


# VALIDASI
# import numpy as np
# import matplotlib.pyplot as plt

# img = np.load("dataset/clean/frame_000.npy")

# print(img.shape)

# plt.imshow(img, cmap="gray")
# plt.show()