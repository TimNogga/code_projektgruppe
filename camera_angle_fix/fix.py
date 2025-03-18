import json
import numpy as np

# Input and output file paths
input_path = "transforms.json"
output_path = "transforms_fixed.json"

# Load original JSON data
with open(input_path, "r") as f:
    data = json.load(f)

# Check if 'camera_angle_x' is missing or incorrect; fix if needed
if "camera_angle_x" not in data or not isinstance(data["camera_angle_x"], (float, int)):
    first_frame = data["frames"][0]
    fl_x = first_frame.get("fl_x", 1000)
    cx = first_frame.get("cx", 500)

    camera_angle_x = 2 * np.arctan(cx / fl_x)
    data["camera_angle_x"] = float(camera_angle_x)
    print(f"Added/fixed camera_angle_x: {camera_angle_x:.6f}")
else:
    print("camera_angle_x already present and valid.")

# Ensure each frame has 'transform_matrix_start'
for frame in data["frames"]:
    if "transform_matrix_start" not in frame:
        frame["transform_matrix_start"] = frame["transform_matrix"]
        print(f"Added missing transform_matrix_start to frame {frame['file_path']}")

# Save modified JSON data
with open(output_path, "w") as f:
    json.dump(data, f, indent=4)

print(f"Successfully saved fixed JSON to {output_path}")
