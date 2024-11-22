import cv2
import numpy as np
from ultralytics import YOLO
from datetime import datetime, timedelta
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import os
import random

# Global variables for drawing lines
lines = []
drawing = False
current_line = []

# Helper Function: Drawing Lines
def draw_line(event, x, y, flags, param):
    global drawing, current_line
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        current_line = [(x, y)]
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        img_copy = param.copy()
        cv2.line(img_copy, current_line[0], (x, y), (255, 255, 255), 2)
        cv2.imshow("Line Drawing", img_copy)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        current_line.append((x, y))
        lines.append(current_line)

# Helper Function: Get Lines
def get_lines(frame, max_lines=None):
    global lines, drawing, current_line
    lines = []
    drawing = False
    current_line = []
    cv2.namedWindow("Line Drawing")
    cv2.setMouseCallback("Line Drawing", draw_line, frame)

    print("Draw the lines of interest (ROIs). Press 'q' when done.")
    while True:
        img_copy = frame.copy()
        for idx, line in enumerate(lines):
            cv2.line(img_copy, line[0], line[1], (100, 205, 90), 2)
            cv2.putText(img_copy, f'L{idx+1}', (line[0][0], line[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("Line Drawing", img_copy)
        key = cv2.waitKey(1)
        if key == ord('q') or (max_lines and len(lines) >= max_lines):
            break

    cv2.destroyWindow("Line Drawing")
    return lines, img_copy

# Helper Function: Generate Line Colors
def generate_line_colors(num_lines):
    return [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(num_lines)]

# Helper Function: Track Objects and Save Results
def track_objects_in_lines(video_path, model_path, lines, frame_skip, recording_start_time):
    model = YOLO(model_path)
    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"

    line_colors = generate_line_colors(len(lines))  # Assign random colors to lines
    tracking_info = {}
    frame_count = 0
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_skip == 0:
            results = model.track(frame, persist=True, conf=0.2)
            elapsed_time = timedelta(seconds=frame_count / fps)
            current_time = recording_start_time + elapsed_time

            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls) if box.cls is not None else None
                        if class_id is None or class_id >= len(model.names):
                            continue
                        track_id = int(box.id) if box.id is not None else None
                        if track_id is None:
                            continue

                        class_name = model.names[class_id]

                        if track_id not in tracking_info:
                            tracking_info[track_id] = {
                                'class': class_name,
                                'lines': {i: {'crossed': False, 'timestamp': 'NIL'} for i in range(len(lines))}
                            }

                        bbox = box.xyxy.cpu().numpy().flatten()
                        bottom_center = ((bbox[0] + bbox[2]) // 2, bbox[3])

                        for i, line in enumerate(lines):
                            if not tracking_info[track_id]['lines'][i]['crossed']:
                                tracking_info[track_id]['lines'][i]['crossed'] = True
                                tracking_info[track_id]['lines'][i]['timestamp'] = current_time.strftime("%Y-%m-%d %H:%M:%S")

                        # Draw bounding box and label on the frame
                        x1, y1, x2, y2 = map(int, bbox)
                        label = f"{class_name} {track_id}"
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Draw the lines on the annotated frame
        for idx, line in enumerate(lines):
            cv2.line(frame, line[0], line[1], line_colors[idx], 2)
            cv2.putText(frame, f"L{idx + 1}", (line[0][0], line[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, line_colors[idx], 2)

        # Show real-time annotated video
        cv2.imshow("Real-Time Detection and Tracking", frame)
        if cv2.waitKey(1) == ord('q'):
            break

        frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    data = []
    for track_id, info in tracking_info.items():
        row = {'Vehicle ID': f"{info['class']}_{track_id}", 'Class': info['class']}
        for i in range(len(lines)):
            row[f'Line {i + 1}'] = 1 if info['lines'][i]['crossed'] else 0
            row[f'Timestamp Line {i + 1}'] = info['lines'][i]['timestamp']
        data.append(row)

    df = pd.DataFrame(data)
    return df

# Custom Input Dialog
def get_recording_info():
    dialog = tk.Toplevel()
    dialog.title("Recording Information")

    tk.Label(dialog, text="Date of Recording (YYYY-MM-DD):").grid(row=0, column=0)
    tk.Label(dialog, text="Time of Recording (HH:MM:SS):").grid(row=1, column=0)
    tk.Label(dialog, text="Latitude:").grid(row=2, column=0)
    tk.Label(dialog, text="Longitude:").grid(row=3, column=0)

    date_entry = tk.Entry(dialog)
    time_entry = tk.Entry(dialog)
    latitude_entry = tk.Entry(dialog)
    longitude_entry = tk.Entry(dialog)

    date_entry.grid(row=0, column=1)
    time_entry.grid(row=1, column=1)
    latitude_entry.grid(row=2, column=1)
    longitude_entry.grid(row=3, column=1)

    def submit():
        dialog.result = {
            "date": date_entry.get(),
            "time": time_entry.get(),
            "latitude": latitude_entry.get(),
            "longitude": longitude_entry.get()
        }
        dialog.destroy()

    tk.Button(dialog, text="Submit", command=submit).grid(row=4, column=0, columnspan=2)
    dialog.wait_window()

    return dialog.result

# Main Function
def main():
    root = tk.Tk()
    root.withdraw()

    recording_info = get_recording_info()
    if not recording_info:
        messagebox.showerror("Error", "Recording information not provided!")
        return

    try:
        recording_start_time = datetime.strptime(f"{recording_info['date']} {recording_info['time']}", "%Y-%m-%d %H:%M:%S")
    except ValueError:
        messagebox.showerror("Error", "Invalid date or time format!")
        return

    video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("Video Files", "*.mp4")])
    if not video_path:
        messagebox.showerror("Error", "No video file selected!")
        return

    model_path = filedialog.askopenfilename(title="Select Model File", filetypes=[("YOLO Model", "*.pt")])
    if not model_path:
        messagebox.showerror("Error", "No model file selected!")
        return

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    assert ret, "Error reading the first frame of the video"
    cap.release()

    print("Draw the lines of interest (ROIs). Press 'q' when done.")
    lines, img_with_lines = get_lines(frame)
    print(f"{len(lines)} lines drawn.")

    save_directory = filedialog.askdirectory(title="Select Directory to Save Results")
    if not save_directory:
        messagebox.showerror("Error", "No directory selected to save data!")
        return

    frame_skip_input = messagebox.askyesno("Frame Skip", "Do you want to skip frames for faster processing?")
    frame_skip = 5 if frame_skip_input else 1

    df = track_objects_in_lines(video_path, model_path, lines, frame_skip, recording_start_time)
    output_path = os.path.join(save_directory, "tracking_results.xlsx")
    df.to_excel(output_path, index=False)
    messagebox.showinfo("Success", f"Tracking results saved at {output_path}.")

if __name__ == "__main__":
    main()
