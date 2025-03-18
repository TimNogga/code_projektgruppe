import cv2
import os

# Define input and output paths
input_folder = "eval"
output_folder = "eval"
angle = 90  # Rotation angle (90, 180, 270)

def rotate_image(image_path, output_path, angle):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    # Get image dimensions
    (h, w) = image.shape[:2]
    
    # Define rotation matrix
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Perform rotation
    rotated = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    # Save rotated image
    cv2.imwrite(output_path, rotated)
    print(f"Rotated image saved to {output_path}")

def rotate_images_in_folder(input_folder, output_folder, angle):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            rotate_image(input_path, output_path, angle)

# Execute rotation
rotate_images_in_folder(input_folder, output_folder, angle)
