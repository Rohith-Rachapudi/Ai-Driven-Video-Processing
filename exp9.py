import cv2
import time

# Open traffic video
cap = cv2.VideoCapture("traffic.mp4")

if not cap.isOpened():
    print("Error: Cannot open traffic.mp4")
    exit()

# Background subtractor
bg = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

frame_count = 0
total_objects = 0
total_time = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_count += 1

    start_time = time.time()

    # Resize frame
    frame = cv2.resize(frame, (800, 600))

    # Background subtraction
    mask = bg.apply(frame)

    # Remove shadows and noise
    _, thresh = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

    # Morphological operation
    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (5, 5)
    )

    thresh = cv2.morphologyEx(
        thresh,
        cv2.MORPH_OPEN,
        kernel
    )

    thresh = cv2.dilate(
        thresh,
        kernel,
        iterations=2
    )

    # Find contours
    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    object_count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very small objects
        if area > 500:

            x, y, w, h = cv2.boundingRect(contour)

            object_count += 1

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display object number
            cv2.putText(
                frame,
                "Object " + str(object_count),
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

    end_time = time.time()

    processing_time = end_time - start_time

    total_time += processing_time
    total_objects += object_count

    # Display information
    cv2.putText(
        frame,
        "Objects: " + str(object_count),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        "Frame: " + str(frame_count),
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 0, 0),
        2
    )

    # Show result
    cv2.imshow(
        "Traffic Object Detection",
        frame
    )

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Release resources
cap.release()
cv2.destroyAllWindows()


# Performance calculation
if frame_count > 0:

    average_time = total_time / frame_count

    fps = 1 / average_time

    average_objects = total_objects / frame_count

    print("\n========== RESULTS ==========")

    print("Total Frames       :", frame_count)

    print(
        "Average Objects    :",
        round(average_objects, 2)
    )

    print(
        "Average Time/Frame :",
        round(average_time, 4),
        "seconds"
    )

    print(
        "Processing FPS     :",
        round(fps, 2)
    )
