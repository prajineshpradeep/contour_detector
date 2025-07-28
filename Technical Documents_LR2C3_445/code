import cv2
import numpy as np
import ezdxf  # Ensure you have this installed with pip install ezdxf

# Function to compactly nest contours on a blank canvas
def compact_nest_contours(contours):
    # Create a blank canvas
    total_width = 0
    max_height = 0
    bounding_boxes = []

    # Calculate total width and maximum height of contours
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append((x, y, w, h))
        total_width += w + 5  # Add spacing
        max_height = max(max_height, h)

    # Create a blank image with the calculated width and height
    nested_image = np.zeros((max_height + 100, total_width + 100, 3), dtype=np.uint8)

    # Track available space for compact nesting
    current_x = 0
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_shifted = contour - contour.min(axis=0)  # Align contour to origin

        # Check and adjust position for no overlap
        while True:
            roi = nested_image[0:max_height, current_x:current_x + w]
            if np.sum(cv2.drawContours(np.zeros_like(roi), [contour_shifted], -1, 255, -1) & roi) == 0:
                break
            current_x += 1

        # Place the contour at the calculated position
        translation_matrix = np.array([[1, 0, current_x], [0, 1, 0]], dtype=np.float32)
        shifted_contour = cv2.transform(contour, translation_matrix)
        cv2.drawContours(nested_image, [shifted_contour], -1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Update the x-coordinate for the next contour
        current_x += w + 5  # Add spacing

    return nested_image

# Function to convert nested polygons to DXF
def export_to_dxf(contours, filename):
    doc = ezdxf.new()  # Create a new DXF document
    msp = doc.modelspace()  # Get the model space
    for contour in contours:
        contour = contour.reshape(-1, 2)  # Reshape contour to a 2D array
        msp.add_lwpolyline(points=contour.tolist(), close=True)
    doc.saveas(filename)  # Save the DXF file

# Initialize camera
cam = cv2.VideoCapture(0)
img_counter = 0
all_contours = []

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break
    cv2.imshow("Camera Feed", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        img_name = f"opencv_frame_{img_counter}.png"
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")

        # Process the captured image for contours
        image = cv2.imread(img_name)
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_gray = cv2.bitwise_not(img_gray)

        # Improved thresholding for better contour detection around shadows
        thresh = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter out small contours based on area
        min_contour_area = 100  # Adjustable based on image
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

        all_contours.extend(filtered_contours)  # Collect filtered contours from all frames
        print(f"Contours from {img_name}: {len(filtered_contours)} contours found.")
        img_counter += 1

# Release camera
cam.release()
cv2.destroyAllWindows()

# Nest all collected contours and export results
if all_contours:
    nested_image = compact_nest_contours(all_contours)

    # Save the nested image as PNG
    nested_image_filename = "compact_nested_output.png"
    cv2.imwrite(nested_image_filename, nested_image)
    print(f"Saved nested image: {nested_image_filename}")

    # Export all nested contours to a DXF file
    export_to_dxf(all_contours, "compact_nested_output.dxf")
    print("Exported to compact_nested_output.dxf")

cv2.destroyAllWindows()
