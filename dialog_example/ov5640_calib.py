import cv2
import numpy as np
import glob
import os

# === Calibration Settings ===
CHECKERBOARD = (5, 6)  # inner corners (rows, columns)
SQUARE_SIZE = 0.025    # meters (25 mm)

# === Prepare object points (3D points in real world) ===
objp = np.zeros((CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = []  # 3D points
imgpoints = []  # 2D points

# === Load all checkerboard images ===
image_paths = glob.glob('calib_images/*.jpg')  # Adjust if using .png
if not image_paths:
    print("No calibration images found in 'calib_images/'")
    exit(1)

for fname in image_paths:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(
            gray, corners, winSize=(11, 11), zeroZone=(-1, -1),
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        )
        imgpoints.append(corners2)

        # Optional: visualize
        img_show = cv2.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
        cv2.imshow('Checkerboard', img_show)
        cv2.waitKey(200)
    else:
        print(f"❌ Corners not found in {fname}")

cv2.destroyAllWindows()

# === Run calibration ===
ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

# === Save calibration data ===
os.makedirs("calibration_output", exist_ok=True)
np.savez("calibration_output/camera_calibration.npz", K=K, dist=dist)

# === Print results ===
print("\n✅ Calibration successful!" if ret else "❌ Calibration failed.")
print("\nCamera matrix (K):")
print(K)
print(f"\nDistortion coefficients:\n{dist.ravel()}")
print(f"\nOptical center (cx, cy): ({K[0,2]:.2f}, {K[1,2]:.2f})")
print(f"Image center (w/2, h/2): ({gray.shape[1]/2:.2f}, {gray.shape[0]/2:.2f})")

# === Undistort test ===
img = cv2.imread(image_paths[0])
h, w = img.shape[:2]
new_K, _ = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), 1, (w, h))

undistorted = cv2.undistort(img, K, dist, None, new_K)
cv2.imshow('Undistorted', undistorted)
cv2.waitKey(0)
cv2.destroyAllWindows()
