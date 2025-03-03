#!/usr/bin/env python3

import json
import numpy as np
import re

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def array_4x4_from_list(lst):
    """Convert a 16-element list into a 4x4 numpy array (row-major)."""
    return np.array(lst, dtype=float).reshape(4, 4)

def rotation_distance_degrees(R1, R2):
    """
    Returns the angle in degrees between two rotation matrices,
    given by the angle of R1^T * R2.
    """
    R = R1.T @ R2
    trace_val = (np.trace(R) - 1.0) / 2.0
    # Clamp for numerical stability
    trace_val = max(min(trace_val, 1.0), -1.0)
    angle_rad = np.arccos(trace_val)
    return np.degrees(angle_rad)

def main():
    # 1) Load the JSON files
    og_data = load_json("og.json")    # your original camera data
    nerf_data = load_json("nerf.json") # your NeRF frames data

    # 2) Build a map from camera_id -> extrinsics
    camera_map = {}
    for cam in og_data["cameras"]:
        cid = cam["camera_id"]
        M_list = cam["extrinsics"]["view_matrix"]
        M = array_4x4_from_list(M_list)
        camera_map[cid] = { "M": M }

    # 3) Build a map from "Cxxxx" -> transform_matrix
    frames_map = {}
    for fr in nerf_data["frames"]:
        filepath = fr["file_path"]  # e.g. "./images/C0000.jpg"
        # Extract the camera ID, e.g. "C0000" from that
        basename = filepath.split('/')[-1]   # "C0000.jpg"
        cam_id = basename.split('.')[0]      # "C0000"
        frames_map[cam_id] = np.array(fr["transform_matrix"], dtype=float)

    # 4) Compare each camera’s M_inv to the corresponding transform_matrix
    all_Q = []
    all_p = []
    matched_ids = []

    for cid, info in camera_map.items():
        if cid not in frames_map:
            continue  # no matching frame
        M = info["M"]
        N = frames_map[cid]

        # invert M
        M_inv = np.linalg.inv(M)

        # Extract rotation & translation from M_inv (Rm, tm) and N (Rn, tn)
        Rm, tm = M_inv[:3, :3], M_inv[:3, 3]
        Rn, tn = N[:3, :3], N[:3, 3]

        # Hypothesize a global rotation Q: Rn ~ Q * Rm
        Q = Rn @ Rm.T
        # Hypothesize p = tn - Q @ tm
        p = tn - Q @ tm

        all_Q.append(Q)
        all_p.append(p)
        matched_ids.append(cid)

    if not all_Q:
        print("No matching cameras found between og.json and nerf.json.")
        return

    # 5) Use the first as a reference
    Q0 = all_Q[0]
    p0 = all_p[0]

    print("=== Global transform reference (from first matched camera) ===")
    print("Q0 (rotation):\n", Q0)
    print("p0 (translation):", p0)
    print()

    for i, cid in enumerate(matched_ids):
        Qi = all_Q[i]
        pi = all_p[i]
        rot_diff = rotation_distance_degrees(Q0, Qi)
        trans_diff = np.linalg.norm(pi - p0)
        print(f"Camera {cid}: rotation diff = {rot_diff:.4f} deg, translation diff = {trans_diff:.4f}")

    print("\nDone! If these diffs are small, it means the same global re-orientation applies to all cameras.")

if __name__ == "__main__":
    main()
import json
import numpy as np
import sys
import re

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def array_4x4_from_list(lst):
    """Convert a 16-element list into a 4x4 numpy array (row-major)."""
    return np.array(lst, dtype=float).reshape(4, 4)

def rotation_distance_degrees(R1, R2):
    """Returns the angle in degrees between two rotation matrices (R1^T * R2)."""
    R = R1.T @ R2
    # The trace-based formula for rotation angle
    trace_val = (np.trace(R) - 1.0) / 2.0
    # Numerical clamping in case of floating errors
    trace_val = max(min(trace_val, 1.0), -1.0)
    angle = np.degrees(np.arccos(trace_val))
    return angle

def main(cameras_json_path, nerf_json_path):
    # 1) Load data
    cameras_data = load_json(cameras_json_path)  # your big cameras JSON
    nerf_data = load_json(nerf_json_path)        # your NeRF frames JSON

    # 2) Build a lookup from camera_id -> extrinsics
    camera_map = {}
    for cam in cameras_data["cameras"]:
        cid = cam["camera_id"]
        M_list = cam["extrinsics"]["view_matrix"]
        M = array_4x4_from_list(M_list)
        camera_map[cid] = { "M": M }

    # 3) Build a lookup from "Cxxxx" to transform_matrix in the NeRF frames
    frames_map = {}
    for fr in nerf_data["frames"]:
        filepath = fr["file_path"]  # e.g. "./images/C0000.jpg"
        # Extract "C0000" from that filename
        basename = filepath.split('/')[-1]   # "C0000.jpg"
        cam_id = basename.split('.')[0]      # "C0000"
        frames_map[cam_id] = np.array(fr["transform_matrix"], dtype=float)

    # 4) For each camera, see if we have a matching transform_matrix
    #    Then invert the camera extrinsics and compare.
    all_Q = []
    all_p = []
    matched_ids = []

    for cid, info in camera_map.items():
        if cid not in frames_map:
            continue  # no matching frame
        M = info["M"]
        N = frames_map[cid]

        # invert M
        M_inv = np.linalg.inv(M)

        # Extract rotation & translation from M_inv and N
        Rm = M_inv[:3, :3]
        tm = M_inv[:3, 3]
        Rn = N[:3, :3]
        tn = N[:3, 3]

        # Suppose there's a global rotation Q s.t. Rn ~ Q * Rm
        # A quick guess is Q = Rn * Rm^T  (if Rm, Rn are nearly pure rotations)
        Q = Rn @ Rm.T

        # Then p = tn - Q @ tm
        p = tn - Q @ tm

        all_Q.append(Q)
        all_p.append(p)
        matched_ids.append(cid)

    if not all_Q:
        print("No matching cameras found between the two files.")
        return

    # 5) Compare them all to the first as a reference
    Q0 = all_Q[0]
    p0 = all_p[0]
    print("=== Global transform reference (from first matched camera) ===")
    print("Q0 (rotation):\n", Q0)
    print("p0 (translation):", p0)
    print()

    for i in range(len(all_Q)):
        cid = matched_ids[i]
        Qi = all_Q[i]
        pi = all_p[i]
        rot_diff = rotation_distance_degrees(Q0, Qi)
        trans_diff = np.linalg.norm(pi - p0)
        print(f"Camera {cid}: rotation diff = {rot_diff:.4f} deg, translation diff = {trans_diff:.4f}")

    print("\nDone. If all the rotation/translation differences are small, it means one global re-orientation is applied.")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <cameras.json> <nerf_frames.json>")
        sys.exit(1)
    cameras_json_path = sys.argv[1]
    nerf_json_path = sys.argv[2]
    main(cameras_json_path, nerf_json_path)
