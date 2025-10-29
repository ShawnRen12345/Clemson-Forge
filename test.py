from ultralytics import YOLO
import cv2
import os

# --- Always work from the script's own folder ---
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# --- File paths (relative to the script folder) ---
image_path = os.path.join(script_dir, "Desk.jpg")
model_path = os.path.join(script_dir, "yolo11x-seg.pt")

# --- Verify both files exist ---
if not os.path.exists(image_path):
    raise FileNotFoundError(f"❌ Image not found in folder: {image_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model not found in folder: {model_path}")

# --- Load YOLO model ---
print("🔄 Loading model...")
model = YOLO(model_path)
print("✅ Model loaded successfully!")

# --- Load image to confirm it's readable ---
img = cv2.imread(image_path)
if img is None:
    raise ValueError("❌ OpenCV cannot read Desk.jpg — check file format or permissions.")
else:
    print(f"✅ Loaded image successfully: {img.shape}")

# --- Run segmentation on the image ---
print("🔍 Running segmentation...")
results = model(img, conf=0.5, save=True)
print("✅ Segmentation complete!")

# --- Process results ---
for result in results:
    num_objects = len(result.boxes)
    print(f"🟢 Number of segmented objects: {num_objects}")

    # Draw segment masks and bounding boxes
    annotated_frame = result.plot()
    annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

    # Add text overlay for object count
    text = f"Objects detected: {num_objects}"
    cv2.putText(
        annotated_frame,
        text,
        (20, 40),  # position
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,  # font scale
        (0, 255, 0),  # green color
        3,  # thickness
        cv2.LINE_AA
    )

    # --- Show the annotated image (if GUI available) ---
    try:
        cv2.imshow("Segmentation Results", annotated_frame)
        print("👀 Press any key in the image window to close it...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except cv2.error:
        print("⚠️ OpenCV display not supported in this environment (e.g., VSCode/Jupyter).")

    # --- Save the annotated image ---
    output_path = os.path.join(script_dir, "Desk_segmented_with_count.jpg")
    cv2.imwrite(output_path, annotated_frame)
    print(f"✅ Saved annotated image to: {output_path}")

    # --- Optional: open the saved image automatically on Windows ---
    if os.name == "nt":  # Windows only
        os.startfile(output_path)
