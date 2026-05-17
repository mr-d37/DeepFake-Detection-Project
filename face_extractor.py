import os
import cv2
import face_recognition
import numpy as np
from pathlib import Path
from tqdm import tqdm


def extract_faces_from_video(video_path, output_dir, num_frames=40):
    """
    Extract faces from a video file and save them as images.

    Args:
        video_path: Path to the input video file
        output_dir: Directory to save extracted face images
        num_frames: Number of frames to extract from the video

    Returns:
        List of paths to saved face images, or None if no faces found
    """
    video_name = Path(video_path).stem
    video_output_dir = os.path.join(output_dir, video_name)
    os.makedirs(video_output_dir, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print(f"Warning: Could not read video {video_path}")
        return None

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    face_paths = []
    saved_count = 0

    for frame_idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()

        if not ret:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_frame, model='hog')

        if face_locations:
            top, right, bottom, left = face_locations[0]

            face_height = bottom - top
            face_width = right - left
            margin = int(0.2 * max(face_height, face_width))

            top = max(0, top - margin)
            bottom = min(frame.shape[0], bottom + margin)
            left = max(0, left - margin)
            right = min(frame.shape[1], right + margin)

            face_crop = rgb_frame[top:bottom, left:right]

            face_resized = cv2.resize(face_crop, (224, 224))
            face_bgr = cv2.cvtColor(face_resized, cv2.COLOR_RGB2BGR)

            face_filename = f"frame_{saved_count:04d}.jpg"
            face_path = os.path.join(video_output_dir, face_filename)
            cv2.imwrite(face_path, face_bgr)

            face_paths.append(face_path)
            saved_count += 1

        if saved_count >= num_frames:
            break

    cap.release()

    if saved_count == 0:
        print(f"No faces detected in {video_path}")
        return None

    return face_paths


def process_video_folder(input_dir, output_dir, label, num_frames=40):
    """
    Process all videos in a folder and extract faces.

    Args:
        input_dir: Directory containing video files
        output_dir: Directory to save extracted faces
        label: 'real' or 'fake' label for the videos
        num_frames: Number of frames to extract per video

    Returns:
        Dictionary mapping video names to list of face image paths
    """
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
    video_files = []

    for ext in video_extensions:
        video_files.extend(Path(input_dir).glob(f"*{ext}"))

    print(f"Processing {len(video_files)} videos from {label} folder...")

    results = {}

    for video_path in tqdm(video_files, desc=f"Extracting {label} faces"):
        face_paths = extract_faces_from_video(
            str(video_path),
            output_dir,
            num_frames=num_frames
        )
        if face_paths:
            results[video_path.stem] = face_paths

    print(f"Successfully extracted faces from {len(results)} videos")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Extract faces from videos')
    parser.add_argument('--input-dir', type=str, required=True,
                        help='Directory containing input videos')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to save extracted faces')
    parser.add_argument('--label', type=str, choices=['real', 'fake'],
                        required=True, help='Label for the videos')
    parser.add_argument('--num-frames', type=int, default=40,
                        help='Number of frames to extract per video')

    args = parser.parse_args()

    process_video_folder(args.input_dir, args.output_dir, args.label, args.num_frames)