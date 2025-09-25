import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
from imageworks.libs.vision.mono import (
    _extract_image_arrays,
    _circular_mean_deg_from_angles,
)


def analyze_image(image_path):
    print(f"\nAnalyzing {Path(image_path).name}")

    # Load and process image
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image: {image_path}")
        return

    # Convert to LAB and HSV
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Extract arrays
    L, a, b = _extract_image_arrays(lab)
    H, S, V = _extract_image_arrays(hsv)

    # Convert hue to degrees
    hue_deg = H * 2  # OpenCV stores hue as 0-180, we want 0-360

    # Sort by luminance
    L_sorted_indices = np.argsort(L)
    L_sorted = L[L_sorted_indices]
    hue_sorted = hue_deg[L_sorted_indices]

    # Calculate mean hue
    mean_hue = _circular_mean_deg_from_angles(hue_deg)

    # Unwrap hue values around mean
    hue_unwrapped = np.copy(hue_sorted)
    for i in range(len(hue_unwrapped)):
        while hue_unwrapped[i] - mean_hue > 180:
            hue_unwrapped[i] -= 360
        while hue_unwrapped[i] - mean_hue < -180:
            hue_unwrapped[i] += 360

    # Try polyfit and catch warning
    with np.warnings.catch_warnings(record=True) as w:
        np.warnings.simplefilter("always")
        coeffs = np.polyfit(L_sorted, hue_unwrapped, 1)
        if len(w) > 0:
            for warning in w:
                print(f"Warning: {warning.message}")

    # Print statistics
    print(f"L range: {L_sorted[0]:.2f} to {L_sorted[-1]:.2f}")
    print(
        f"Hue range (unwrapped): {hue_unwrapped.min():.2f} to {hue_unwrapped.max():.2f}"
    )
    print(f"Number of unique L values: {len(np.unique(L_sorted))}")
    print(f"Number of unique hue values: {len(np.unique(hue_unwrapped))}")

    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.scatter(L_sorted, hue_unwrapped, alpha=0.1, s=1)
    plt.plot(
        L_sorted, np.polyval(coeffs, L_sorted), "r-", label=f"Slope: {coeffs[0]:.4f}"
    )
    plt.xlabel("Luminance")
    plt.ylabel("Unwrapped Hue (degrees)")
    plt.title("Hue vs Luminance")
    plt.legend()

    plt.subplot(122)
    plt.hist2d(L_sorted, hue_unwrapped, bins=50)
    plt.xlabel("Luminance")
    plt.ylabel("Unwrapped Hue (degrees)")
    plt.title("Density Plot")
    plt.colorbar()

    plt.tight_layout()
    plt.savefig(f"polyfit_analysis_{Path(image_path).stem}.png")
    plt.close()


# Analyze both problematic images
images = ["01_036_SerialPrint1_2425.jpg", "01_002_SerialPrint3_2425.jpg"]

for img_path in images:
    analyze_image(img_path)
