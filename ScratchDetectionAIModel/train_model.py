from keras import layers, models, applications
import os

from data_processing import preprocess_data

# trained model file
trained_model_file = "trained_model.keras"

# Step 1: Preprocess the data
bounding_boxes_file = "bounding_boxes.txt"
images, annotations = preprocess_data(bounding_boxes_file)

# Step 2: Define or load the model architecture
if os.path.exists(trained_model_file):
    model = models.load_model(trained_model_file)
    print("Loaded existing model.")
else:
    # Define the model architecture
    base_model = applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = layers.GlobalAveragePooling2D()(base_model.output)
    x = layers.Dense(1024, activation='relu')(x)
    output = layers.Dense(4, activation='sigmoid')(x)  # Output layer for bounding box coordinates
    model = models.Model(inputs=base_model.input, outputs=output)
    print("Created new model.")

# Step 3: Compile the model
model.compile(optimizer='adam', loss='mse')

# Step 4: Train the model with data augmentation
# Add data augmentation techniques here (e.g., rotation, flipping, etc.)
# You can use data augmentation utilities provided by Keras (ImageDataGenerator)
# Model training code goes here...
model.fit(images, annotations, validation_split=0.1, epochs=1000, batch_size=5)

# Step 5: Save the trained model
model.save(trained_model_file)
