import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

clean_dir = "dataset/clean"
race_dir = "dataset/race"   # pilih salah satu run
comp_dir = "dataset/compensated_residual"

psnr_race_list = []
psnr_comp_list = []

ssim_race_list = []
ssim_comp_list = []

for i in range(150):

    gt = np.load(f"{clean_dir}/frame_{i:03d}.npy")
    race = np.load(f"{race_dir}/frame_{i:03d}_run2.npy")
    comp = np.load(f"{comp_dir}/frame_{i:03d}.npy")

    print(race.min(), race.max())
    print(comp.min(), comp.max())
    print(gt.min(), gt.max())

    p_race = psnr(gt, race, data_range=255)
    p_comp = psnr(gt, comp, data_range=255)

    s_race = ssim(gt, race, data_range=255)
    s_comp = ssim(gt, comp, data_range=255)

    psnr_race_list.append(p_race)
    psnr_comp_list.append(p_comp)

    ssim_race_list.append(s_race)
    ssim_comp_list.append(s_comp)

print("==== RESULTS ====")

print("PSNR Race      :", np.mean(psnr_race_list))
print("PSNR Comp      :", np.mean(psnr_comp_list))

print("SSIM Race      :", np.mean(ssim_race_list))
print("SSIM Comp      :", np.mean(ssim_comp_list))