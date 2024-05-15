import cv2
import numpy as np

def preprocess_data(bounding_boxes_file, target_size=(224, 224)):
    # Read bounding boxes file
    with open(bounding_boxes_file, 'r') as f:
        lines = f.readlines()

    # Initialize lists to store images and corresponding bounding box annotations
    images = []
    annotations = []

    for line in lines[1:]:  # Skip the header line
        parts = line.strip().split()
        image_path = parts[0]
        bbox = list(map(int, parts[1:]))

        # Read the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Failed to load image:", image_path)
            continue

        # Resize the image while preserving aspect ratio
        image_resized = cv2.resize(image, target_size)
        image_normalized = image_resized / 255.0  # Normalize pixel values

        # Adjust bounding box coordinates
        h_ratio = target_size[0] / image.shape[0]
        w_ratio = target_size[1] / image.shape[1]
        bbox[0] = int(bbox[0] * w_ratio)  # x-coordinate
        bbox[1] = int(bbox[1] * h_ratio)  # y-coordinate
        bbox[2] = int(bbox[2] * w_ratio)  # width
        bbox[3] = int(bbox[3] * h_ratio)  # height

        images.append(image_normalized)
        annotations.append(bbox)

    return np.array(images), np.array(annotations)
