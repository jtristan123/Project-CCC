import cv2
import os

# === Settings ===
save_dir = "captured_images"
os.makedirs(save_dir, exist_ok=True)

cam = cv2.VideoCapture(0)  # USB webcam
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 896)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 504)

img_count = 0
print("Press SPACE to capture image, ESC to exit.")

while True:
    ret, frame = cam.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Live Feed - Press SPACE to Save", frame)
    key = cv2.waitKey(1)

    if key % 256 == 27:  # ESC pressed
        print("Escape hit, closing...")
        break
    elif key % 256 == 32:  # SPACE pressed
        img_name = f"{save_dir}/image_{img_count:04}.jpg"
        cv2.imwrite(img_name, frame)
        print(f"Saved {img_name}")
        img_count += 1

cam.release()
cv2.destroyAllWindows()
