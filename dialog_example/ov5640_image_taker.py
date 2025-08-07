import cv2
import os

# === Configuration ===
camera_index = 2  # Change to 1, 2, etc., if needed
output_dir = "calib_images"
os.makedirs(output_dir, exist_ok=True)

# === Initialize camera ===
cap = cv2.VideoCapture(camera_index)

if not cap.isOpened():
    print(f"❌ Failed to open camera at index {camera_index}")
    exit(1)

print("📷 Press SPACE to capture, ESC to exit.")

img_count = len(os.listdir(output_dir))  # Continue from existing files

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame")
        break

    cv2.imshow("Camera", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        print("👋 Exiting...")
        break
    elif key == 32:  # SPACE
        filename = f"img_{img_count:03d}.jpg"
        path = os.path.join(output_dir, filename)
        cv2.imwrite(path, frame)
        print(f"✅ Saved: {path}")
        img_count += 1

cap.release()
cv2.destroyAllWindows()
