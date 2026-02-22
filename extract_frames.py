import cv2
import os

video_folder = "videos"
output_root = "frames_raw"
fps_extract = 5  # images par seconde

os.makedirs(output_root, exist_ok=True)

video_files = [f for f in os.listdir(video_folder)
               if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]

video_files.sort()

vid_counter = 1

for video_name in video_files:

    video_path = os.path.join(video_folder, video_name)

    vid_id = f"VID_{vid_counter:04d}"
    video_output_folder = os.path.join(output_root, f"{vid_id}")
    os.makedirs(video_output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)

    video_fps = cap.get(cv2.CAP_PROP_FPS)

    if video_fps == 0:
        print(f"⚠️ Impossible de lire le FPS de {video_name}, skip.")
        continue

    step = max(1, int(video_fps / fps_extract))

    frame_idx = 0
    saved = 0

    print(f"\nTraitement de {video_name} → {vid_id}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % step == 0:
            filename = f"{vid_id}_{saved:06d}.jpg"
            cv2.imwrite(os.path.join(video_output_folder, filename), frame)
            saved += 1

        frame_idx += 1

    cap.release()

    print(f"{saved} images extraites.")
    vid_counter += 1

print("\nExtraction terminée.")