import cv2
import os

# Function to draw bounding box
def draw_bbox(event, x, y, flags, param):
    global ix, iy, drawing, mode, bbox_list

    print(f"Event: {event}, x: {x}, y: {y}, flags: {flags}, param: {param}")

    if event == cv2.EVENT_LBUTTONDOWN:
        print("Left mouse button down")
        drawing = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        print("Mouse move")
        if drawing == True:
            # Reset image to its original state
            image[:] = clone[:]
            # Draw a rectangle from the starting point to the current mouse position
            cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)

    elif event == cv2.EVENT_LBUTTONUP:
        print("Left mouse button up")
        drawing = False
        cv2.rectangle(image, (ix, iy), (x, y), (0, 255, 0), 2)
        bbox = (min(ix, x), min(iy, y), abs(ix - x), abs(iy - y))
        bbox_list.append(bbox)
        cv2.imshow('image', image)

# Folder containing images (relative path)
folder_path = "./dataset_images"

# Get list of image files in folder
image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

# Open a text file to store bounding box data
with open("bounding_boxes.txt", "w") as f:
    f.write("Bounding boxes")

# Loop through images
for image_file in image_files:
    print(f"Processing image: {image_file}")
    # Read image
    image_path = os.path.join(folder_path, image_file)
    image = cv2.imread(image_path)
    clone = image.copy()

    # Initialize variables for bounding box drawing
    drawing = False
    ix, iy = -1, -1
    bbox_list = []

    # Show image
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_bbox)

    while True:
        cv2.imshow('image', image)
        key = cv2.waitKey(1) & 0xFF

        # Press 'r' to reset bounding box
        if key == ord('r'):
            print("Resetting bounding box")
            image = clone.copy()

        # Press 'c' to confirm bounding box and move to next image
        elif key == ord('c'):
            print("Confirming bounding box")
            # Save image with bounding box
            cv2.imwrite(os.path.join("annotated_images", image_file), image)

            # Save bounding box coordinates
            with open("bounding_boxes.txt", "a") as f:
                if len(bbox_list) == 0:
                    # If no bounding box added, write a special label indicating no scratches
                    f.write(f"{image_path} -1 -1 -1 -1\n")
                else:
                    for bbox in bbox_list:
                        label = " ".join(map(str, bbox))  # Convert tuple to string
                        f.write(f"{image_path} {label}\n")
            break

        # Press 'q' to quit
        elif key == ord('q'):
            print("Exiting")
            exit()

    cv2.destroyAllWindows()
