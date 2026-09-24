import cv2
import os

# ---------------------------------------
# Step 1: Ask for the video file path
# ---------------------------------------
video_path = input("Enter the video file path: ").strip()

# Remove quotes if the user pasted a quoted path
video_path = video_path.strip('"').strip("'")

# ---------------------------------------
# Step 2: Check if file exists
# ---------------------------------------
if not os.path.isfile(video_path):
    print("\nError: Video file not found.")
    print("Please check the file path and try again.")
    exit()

# ---------------------------------------
# Step 3: Load the video file
# ---------------------------------------
cap = cv2.VideoCapture(video_path)

# Check whether video is opened successfully
if not cap.isOpened():
    print("\nError: Could not open the video.")
    exit()

# ---------------------------------------
# Step 4: Extract video properties
# ---------------------------------------
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# ---------------------------------------
# Step 5: Calculate video duration
# ---------------------------------------
if fps > 0:
    duration = total_frames / fps
else:
    duration = 0

minutes = int(duration // 60)
seconds = int(duration % 60)

# ---------------------------------------
# Step 6: Display video properties
# ---------------------------------------
print("\n========== VIDEO PROPERTIES ==========")
print("Video File       :", video_path)
print("Frame Width      :", frame_width, "pixels")
print("Frame Height     :", frame_height, "pixels")
print("FPS              :", fps)
print("Total Frames     :", total_frames)
print("Video Duration   :", f"{minutes} minutes {seconds} seconds")
print("=======================================")

print("\nPlaying video...")
print("Press 'q' to exit.")

# ---------------------------------------
# Step 7: Display video frame-by-frame
# ---------------------------------------
while True:

    ret, frame = cap.read()

    # Stop if video ends
    if not ret:
        break

    # Display frame
    cv2.imshow("Video - OpenCV", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------------------
# Step 8: Release resources
# ---------------------------------------
cap.release()
cv2.destroyAllWindows()

print("\nVideo processing completed.")