import os
import cv2
import numpy as np
from keras import models
import tensorflow as tf

# Load the pre-trained model
model = models.load_model('trained_model.keras', compile=True)

# Directory containing the dataset images
dataset_dir = 'dataset_images'

# Define the loss function
loss_fn = tf.keras.losses.MeanSquaredError()

# Define the optimizer
optimizer = tf.keras.optimizers.Adam()

# Create the training step function
@tf.function
def train_step(image_batch, annotation_batch):
    with tf.GradientTape() as tape:
        predictions = model(image_batch)
        loss = loss_fn(annotation_batch, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def draw_rectangle(event, x, y, flags, param):
    global bbox, mode

    if event == cv2.EVENT_LBUTTONDOWN:
        bbox = [(x, y)]  # Store the starting position of the bounding box

    elif event == cv2.EVENT_LBUTTONUP:
        bbox.append((x, y))  # Store the ending position of the bounding box
        cv2.rectangle(image_with_box, bbox[0], bbox[1], (0, 255, 0), 2)
        cv2.imshow('Image with Bounding Box', image_with_box)

        # Set mode to 'draw' when the user confirms the predicted bounding box
        mode = 'draw'

# Main loop
for filename in os.listdir(dataset_dir):
    if filename.endswith('.jpg'):
        image_path = os.path.join(dataset_dir, filename)

        # Load the image
        image = cv2.imread(image_path)
        image_with_box = image.copy()

        # Use the model to predict the bounding box
        # Resize the image to match the model's input shape
        resized_image = cv2.resize(image, (224, 224))
        predicted_bbox = model.predict(np.array([resized_image]))[0]
        xmin, ymin, xmax, ymax = map(int, predicted_bbox)  # Ensure integer coordinates

        # Draw the predicted bounding box
        cv2.rectangle(image_with_box, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        # Display the image with the predicted bounding box
        cv2.imshow('Image with Predicted Bounding Box', image_with_box)

        # Set up mouse event handling
        bbox = [(0, 0), (0, 0)]
        mode = 'predict'
        cv2.setMouseCallback('Image with Predicted Bounding Box', draw_rectangle)

        # Wait for user input
        while True:
            key = cv2.waitKey(1) & 0xFF
            if mode == 'draw':
                # User provided corrected bounding box
                # Convert corrected_bbox to the format (xmin, ymin, xmax, ymax)
                corrected_bbox = (min(bbox[0][0], bbox[1][0]), min(bbox[0][1], bbox[1][1]),
                                max(bbox[0][0], bbox[1][0]), max(bbox[0][1], bbox[1][1]))
                # Update annotations with corrected_bbox and retrain model
                annotations = np.array([corrected_bbox])
                print(annotations)
                # Retrain the model with updated annotations
                loss = train_step(np.array([resized_image]), np.array([annotations]))
                print("Loss:", loss.numpy())
                # Save the retrained model
                model.save('trained_model.keras')
                print("Saving new model")
                mode = 'predict'
                break
            elif key == ord('c'):
                # User confirms predicted bounding box
                break
            elif key == 27:
                # Press Esc to exit
                cv2.destroyAllWindows()
                exit()

cv2.destroyAllWindows()
