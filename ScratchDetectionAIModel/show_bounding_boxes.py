import cv2
import os
import numpy as np

def load_bounding_boxes(file_path):
    bounding_boxes = {}
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            line = line.strip().split()
            if len(line) == 5:  # Assuming format is: image_path x y width height
                image_path = line[0]
                x, y, width, height = map(int, line[1:])
                bounding_boxes.setdefault(image_path, []).append((x, y, width, height))
    return bounding_boxes

def draw_bounding_boxes(image, boxes, original_shape, new_shape):
    # Calculate scale factors
    width_scale = new_shape[1] / original_shape[1]
    height_scale = new_shape[0] / original_shape[0]

    for box in boxes:
        x, y, width, height = box
        # Scale the bounding box coordinates
        x = int(x * width_scale)
        y = int(y * height_scale)
        width = int(width * width_scale)
        height = int(height * height_scale)
        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return image


def preprocess_image(image):
    original_shape = image.shape[:2]  # Get the original shape (height, width)
    # Your preprocessing steps here
    # For example:
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    return image, original_shape


def main():
    bounding_boxes_file = "bounding_boxes.txt"
    
    bounding_boxes = load_bounding_boxes(bounding_boxes_file)
    
    for image_path, boxes in bounding_boxes.items():
        if os.path.exists(image_path):
            image = cv2.imread(image_path)
            if image is not None:
                preprocessed_image, original_shape = preprocess_image(image.copy())
                image_with_boxes = draw_bounding_boxes(preprocessed_image, boxes, original_shape, preprocessed_image.shape[:2])
                cv2.imshow("Image with Bounding Boxes", image_with_boxes)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print(f"Failed to load image: {image_path}")
        else:
            print(f"Image file not found: {image_path}")

if __name__ == "__main__":
    main()
