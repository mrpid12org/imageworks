import numpy as np
import cv2
from pathlib import Path


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
    L = lab[:, :, 0]  # L channel from LAB
    H = hsv[:, :, 0]  # H channel from HSV

    # Convert hue to degrees and analyze
    hue_deg = H.astype(float) * 2  # OpenCV stores hue as 0-180, we want 0-360

    # Flatten arrays
    L = L.flatten()
    hue_deg = hue_deg.flatten()

    # Sort by luminance
    L_sorted_indices = np.argsort(L)
    L_sorted = L[L_sorted_indices]
    hue_sorted = hue_deg[L_sorted_indices]

    # Calculate mean hue (circular)
    hue_rad = np.radians(hue_sorted)
    mean_cos = np.mean(np.cos(hue_rad))
    mean_sin = np.mean(np.sin(hue_rad))
    mean_hue = np.degrees(np.arctan2(mean_sin, mean_cos))
    if mean_hue < 0:
        mean_hue += 360

    # Unwrap hue values around mean
    hue_unwrapped = np.copy(hue_sorted)
    for i in range(len(hue_unwrapped)):
        while hue_unwrapped[i] - mean_hue > 180:
            hue_unwrapped[i] -= 360
        while hue_unwrapped[i] - mean_hue < -180:
            hue_unwrapped[i] += 360

    # Try polyfit and catch warning
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        coeffs = np.polyfit(L_sorted, hue_unwrapped, 1)
        if len(w) > 0:
            for warning in w:
                print(f"Warning: {warning.message}")

    # Print analysis
    print(f"Image shape: {img.shape}")
    print(f"L range: {L_sorted[0]:.2f} to {L_sorted[-1]:.2f}")
    print(
        f"Hue range (unwrapped): {hue_unwrapped.min():.2f} to {hue_unwrapped.max():.2f}"
    )
    print(f"Mean hue: {mean_hue:.2f}")
    print(f"Number of unique L values: {len(np.unique(L_sorted))}")
    print(f"Number of unique hue values: {len(np.unique(hue_unwrapped))}")
    print(
        f"Condition number of polyfit: {np.linalg.cond(np.vstack([L_sorted, np.ones_like(L_sorted)]).T):.2e}"
    )
    print(f"Slope from polyfit: {coeffs[0]:.4f}")

    # Additional diagnostics
    num_samples = len(L_sorted)
    print(f"Number of pixels: {num_samples}")
    print(f"L standard deviation: {np.std(L_sorted):.2f}")
    print(f"Hue standard deviation: {np.std(hue_unwrapped):.2f}")

    # Check for potential causes of poor conditioning
    l_range = L_sorted[-1] - L_sorted[0]
    print(f"L value range: {l_range:.2f}")
    if l_range < 1.0:
        print("WARNING: Very small luminance range - could cause poor conditioning")

    # Count clustered values
    l_hist, _ = np.histogram(L_sorted, bins=50)
    max_cluster = np.max(l_hist) / num_samples * 100
    print(f"Maximum percentage of L values in single bin: {max_cluster:.1f}%")
    if max_cluster > 50:
        print("WARNING: Large cluster of similar L values detected")


# Analyze both problematic images
images = ["01_036_SerialPrint1_2425.jpg", "01_002_SerialPrint3_2425.jpg"]

for img_path in images:
    analyze_image(img_path)
