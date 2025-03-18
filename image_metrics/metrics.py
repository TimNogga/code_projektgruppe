import os
import cv2
import torch
import lpips
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Device configuration (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LPIPS model initialization
lpips_model = lpips.LPIPS(net='vgg').to(device)


def compute_metrics(img1, img2):
    # PSNR calculation (pixel-wise fidelity)
    psnr_value = cv2.PSNR(img1, img2)

    # SSIM calculation (structural similarity, grayscale)
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    ssim_value = ssim(img1_gray, img2_gray, data_range=img2_gray.max() - img2_gray.min(), win_size=3)

    # LPIPS calculation (deep perceptual similarity)
    img1_torch = torch.tensor(img1, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    img2_torch = torch.tensor(img2, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device) / 255.0
    lpips_value = lpips_model(img1_torch, img2_torch).item()

    return psnr_value, ssim_value, lpips_value


def evaluate_against_ref(test_dir, ref_dir):
    results = []
    test_files = sorted([f for f in os.listdir(test_dir) if f.endswith('.jpg')])

    for filename in test_files:
        ref_path = os.path.join(ref_dir, filename)
        test_path = os.path.join(test_dir, filename)

        if os.path.exists(ref_path) and os.path.exists(test_path):
            ref_img = cv2.imread(ref_path)
            test_img = cv2.imread(test_path)

            if ref_img.shape != test_img.shape:
                test_img = cv2.resize(test_img, (ref_img.shape[1], ref_img.shape[0]))

            psnr_val, ssim_val, lpips_val = compute_metrics(ref_img, test_img)
            results.append((filename, psnr_val, ssim_val, lpips_val))
            print(f"{filename:15} PSNR: {psnr_val:.4f}  SSIM: {ssim_val:.4f}  LPIPS: {lpips_val:.4f}")

    if results:
        avg_psnr = np.mean([r[1] for r in results])
        avg_ssim = np.mean([r[2] for r in results])
        avg_lpips = np.mean([r[3] for r in results])

        print(f"\nSummary for {os.path.basename(test_dir)}:")
        print(f"Avg PSNR  = {avg_psnr:.4f}")
        print(f"Avg SSIM  = {avg_ssim:.4f}")
        print(f"Avg LPIPS = {avg_lpips:.4f}")

        return avg_psnr, avg_ssim, avg_lpips
    else:
        print("No matching images found.")
        return None, None, None


if __name__ == '__main__':
    ref_dir = "eval"
    test_dirs = {
        "novel_16views": "eval",
    }

    results_summary = {}
    for label, test_dir in test_dirs.items():
        avg_psnr, avg_ssim, avg_lpips = evaluate_against_ref(test_dir, ref_dir)
        results_summary[label] = {"PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips}

    print("\nOverall Summary:")
    for label, metrics in results_summary.items():
        print(f"{label}: PSNR: {metrics['PSNR']:.4f}, SSIM: {metrics['SSIM']:.4f}, LPIPS: {metrics['LPIPS']:.4f}")
