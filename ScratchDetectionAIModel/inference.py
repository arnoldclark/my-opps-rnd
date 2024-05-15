import cv2
import numpy as np
from keras import models

# Load the trained model
model = models.load_model('trained_model.keras', compile=False)  # Load model without compiling

# Load a test image
test_image_path = "test_image2.jpg"
test_image = cv2.imread(test_image_path)
test_image_resized = cv2.resize(test_image, (224, 224))
test_image_normalized = test_image_resized / 255.0  # Normalize pixel values

# Perform inference to predict bounding box annotations
predictions = model.predict(np.array([test_image_normalized]))

# Convert predictions from normalized coordinates to pixel coordinates
predicted_bboxes = []
for pred in predictions:
    xmin = int(pred[0] * test_image_resized.shape[1])
    ymin = int(pred[1] * test_image_resized.shape[0])
    xmax = int(pred[2] * test_image_resized.shape[1])
    ymax = int(pred[3] * test_image_resized.shape[0])
    predicted_bboxes.append((xmin, ymin, xmax, ymax))

print("Predicted bounding boxes:", predicted_bboxes)

# Draw the bounding box on the image
color = (0, 255, 0)  # Green color
thickness = 2
for bbox in predicted_bboxes:
    cv2.rectangle(test_image_resized, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

# Display the image with the bounding box
cv2.imshow("Image with Predicted Bounding Box", test_image_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
