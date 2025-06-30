import cv2
import numpy as np

def add_gaussian_noise(image, mean=0, sigma=25):
    """
    Add Gaussian noise to an image.

    Parameters:
    - image: Input image (numpy array).
    - mean: Mean of the Gaussian noise.
    - sigma: Standard deviation of the Gaussian noise.

    Returns:
    - Noisy image (numpy array).
    """
    row, col, ch = image.shape
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = image + gauss
    noisy = np.clip(noisy, 0, 255)  # Ensure pixel values are in the valid range [0, 255]
    noisy = noisy.astype(np.uint8)
    return noisy

# Read an image from file
img_paths = ['/scratch/students/2023-fall-sp-liying/dataset/SID/Sony_visrgb/00029_00_10s.png']
for img_path in img_paths:
    image = cv2.imread(img_path)
    noisy_image = add_gaussian_noise(image, sigma=50)
    cv2.imwrite(img_path.replace('.png', '_noised.png'), noisy_image)

