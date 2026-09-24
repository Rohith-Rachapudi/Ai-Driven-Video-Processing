import cv2
import os

# ============================================================
# 1. INPUT VIDEO
# ============================================================

input_video = r"C:\Users\Rohith\Desktop\college\AI driven Video processing\highway.mp4"
compressed_video = "compressed.mp4"
analytics_video = "analytics.mp4"

# Check whether input video exists
if not os.path.exists(input_video):
    print("Error: Input video not found:", input_video)
    exit()

# Open input video
cap = cv2.VideoCapture(input_video)

if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30.0

# ============================================================
# 2. BACKGROUND SUBTRACTION
# ============================================================

background_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

# ============================================================
# 3. OUTPUT VIDEO
# ============================================================

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    analytics_video,
    fourcc,
    fps,
    (width, height)
)

if not out.isOpened():
    print("Error: Could not create output video.")
    cap.release()
    exit()

# ============================================================
# 4. MOTION DETECTION
# ============================================================

print("Motion detection started...")

kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (5, 5)
)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Apply background subtraction
    mask = background_subtractor.apply(frame)

    # Remove noise
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_DILATE,
        kernel
    )

    # Find moving objects
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    motion_count = 0

    for contour in contours:

        area = cv2.contourArea(contour)

        # Ignore very small objects/noise
        if area < 500:
            continue

        x, y, w, h = cv2.boundingRect(contour)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

        motion_count += 1

    # Display motion count
    cv2.putText(
        frame,
        "Moving Objects: " + str(motion_count),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2
    )

    # Write analytics video
    out.write(frame)

    # Display
    cv2.imshow(
        "Smart City CCTV Analytics",
        frame
    )

    # Press Q to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================
# 5. RELEASE RESOURCES
# ============================================================

cap.release()
out.release()
cv2.destroyAllWindows()

# ============================================================
# 6. FILE SIZE ANALYSIS
# ============================================================

try:
    original_size = os.path.getsize(input_video)
    analytics_size = os.path.getsize(analytics_video)

    if original_size > 0:
        storage_reduction = (
            (original_size - analytics_size)
            / original_size
        ) * 100

        compression_ratio = (
            original_size / analytics_size
            if analytics_size > 0
            else 0
        )
    else:
        storage_reduction = 0
        compression_ratio = 0

except OSError:
    storage_reduction = 0
    compression_ratio = 0

# ============================================================
# 7. FINAL RESULTS
# ============================================================

print("\n==========================================")
print("        EXPERIMENT COMPLETED")
print("==========================================")

print("\nOriginal Video:")
print(input_video)

print("\nCompressed Video:")
print(compressed_video)

print("\nAnalytics Video:")
print(analytics_video)

print(
    "\nStorage Reduction:",
    round(storage_reduction, 2),
    "%"
)

print(
    "Compression Ratio:",
    round(compression_ratio, 2),
    ":1"
)

print("\nVideo compression and analytics completed successfully.")