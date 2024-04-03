import numpy as np
import cv2
import os


def denoise_salt_and_pepper(image, kernel_size=3, threshold=0.5):
    # Extracting the height and width of the image
    height, width = image.shape

    # Creating a copy of the input image to store the denoised result
    new_image = np.copy(image)

    # Calculating the padding required for the filter kernel
    pad = kernel_size // 2

    # Iterating through each pixel of the image
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Extracting a patch of pixels centered at the current pixel
            patch = image[i - pad : i + pad + 1, j - pad : j + pad + 1]

            # Checking if a random value is less than the threshold
            if np.random.rand() < threshold:
                # Setting the current pixel value to the minimum value within the patch
                new_image[i, j] = np.min(patch)
            # Checking if another random value is less than the threshold
            elif np.random.rand() < threshold:
                # Setting the current pixel value to the maximum value within the patch
                new_image[i, j] = np.max(patch)

    # Returning the denoised image
    return new_image


def super_resolution(image, factor=2):
    # Extracting the height and width of the input image
    height, width = image.shape

    # Calculating the new height and width after upscaling
    new_height, new_width = height * factor, width * factor

    # Creating a new image with zeros to store the upscaled result
    new_image = np.zeros((new_height, new_width), dtype=np.uint8)

    # Iterating through each pixel of the original image
    for i in range(height):
        for j in range(width):
            # Copying the pixel value to corresponding locations in the upscaled image
            new_image[
                i * factor : i * factor + factor, j * factor : j * factor + factor
            ] = image[i, j]

    # Returning the upscaled image
    return new_image


def smoothening(image, kernel_size=3):
    # Extracting the height and width of the input image
    height, width = image.shape

    # Creating a copy of the input image to store the smoothed result
    new_image = np.copy(image)

    # Calculating the padding required for the filter kernel
    pad = kernel_size // 2

    # Generating the averaging filter kernel
    kernel = np.ones((kernel_size, kernel_size), dtype=np.float32) / (kernel_size**2)

    # Iterating through each pixel of the image
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Extracting a patch of pixels centered at the current pixel
            patch = image[i - pad : i + pad + 1, j - pad : j + pad + 1]

            # Applying the filter kernel to the patch and summing the results
            new_image[i, j] = np.sum(patch * kernel)

    # Converting the result to uint8 data type
    return new_image.astype(np.uint8)


def convolution(image, kernel):
    # Extracting the height and width of the input image
    height, width = image.shape

    # Extracting the size of the kernel
    kernel_size = kernel.shape[0]

    # Calculating the padding required for the kernel
    pad = kernel_size // 2

    # Initializing a new image to store the convolved result
    new_image = np.zeros((height, width), dtype=np.uint8)

    # Iterating through each pixel of the image
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Extracting a patch of pixels centered at the current pixel
            patch = image[i - pad : i + pad + 1, j - pad : j + pad + 1]

            # Applying the kernel to the patch and summing the results
            new_image[i, j] = np.sum(patch * kernel)

    # Converting the result to uint8 data type
    return new_image.astype(np.uint8)


def sharpening(image, alpha=1.5):
    # Define a sharpening kernel
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])

    # Convolve the image with the sharpening kernel
    blurred = convolution(image, sharpen_kernel)

    # Blend the original image with the sharpened image to enhance edges and details
    sharpened = cv2.addWeighted(image, alpha, blurred, 1 - alpha, 0)

    return sharpened


def gaussian_blur(image, kernel_size=5):
    # Get the height and width of the image
    height, width = image.shape

    # Create a new image to store the blurred result
    new_image = np.copy(image)

    # Calculate padding for the kernel
    pad = kernel_size // 2

    # Initialize a kernel matrix
    kernel = np.zeros((kernel_size, kernel_size))

    # Define the standard deviation for Gaussian distribution
    sigma = 1.0

    # Fill the kernel matrix with Gaussian values
    for i in range(-pad, pad + 1):
        for j in range(-pad, pad + 1):
            # Calculate the Gaussian value for each position in the kernel
            kernel[i + pad, j + pad] = np.exp(-(i**2 + j**2) / (2 * sigma**2))

    # Normalize the kernel to ensure the sum of its elements is 1
    kernel /= np.sum(kernel)

    # Apply the Gaussian blur to the image
    for i in range(pad, height - pad):
        for j in range(pad, width - pad):
            # Extract the patch from the image
            patch = image[i - pad : i + pad + 1, j - pad : j + pad + 1]
            # Convolve the patch with the Gaussian kernel and store the result
            new_image[i, j] = np.sum(patch * kernel)

    # Convert the result to uint8 data type
    return new_image.astype(np.uint8)


def apply_augmentations(image_path):
    def rotate(image):
        # Randomly select rotation angle between -30 and 30 degrees
        angle = np.random.uniform(-30, 30)
        height, width = image.shape[:2]
        # Compute rotation matrix to rotate around the center of the image
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

    def zoom(image):
        scale_factor = np.random.uniform(0.8, 1.2)
        height, width = image.shape[:2]
        # Compute new height and width after zooming
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        return cv2.resize(image, (new_width, new_height))

    def speedup(image):
        # Resize image with speed factor
        speed_factor = np.random.uniform(0.5, 2.0)
        return cv2.resize(image, None, fx=speed_factor, fy=speed_factor)

    # Define list of augmentation functions
    aug_functions = [rotate, zoom, speedup]
    np.random.shuffle(aug_functions)

    counter = 0
    original_image = cv2.imread(image_path)

    for fun in aug_functions:
        if np.random.rand() < 0.5:
            original_image = fun(original_image)
            counter += 1

    # If at least one augmentation was applied, overwrite the original image file
    if counter > 0:
        cv2.imwrite(image_path, original_image)


def guassian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size : size + 1, -size : size + 1]
    normal = 1 / (2.0 * np.pi * sigma**2)
    g = np.exp(-((x**2 + y**2) / (2.0 * sigma**2))) * normal
    return g


image_dir = "Dataset/"
from tqdm import tqdm

for root, dirs, files in os.walk(image_dir):
    for filename in tqdm(files):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(root, filename)
            image = cv2.imread(image_path)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img = cv2.filter2D(img, -1, guassian_kernel(27, 1))
            img = sharpening(img, alpha=1)
            img = smoothening(img, kernel_size=3)
            img = super_resolution(img, factor=2)
            cv2.imwrite(image_path, img)
