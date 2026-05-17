import os
import argparse
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import VideoInferenceDataset
from model import create_model


def predict(model, dataloader, device):
    """
    Run inference on video frames.

    Args:
        model: Trained DeepFakeDetector model
        dataloader: DataLoader for inference data
        device: Device to run inference on

    Returns:
        Prediction probabilities and predicted class
    """
    model.eval()

    all_probs = []

    with torch.no_grad():
        for frames in dataloader:
            frames = frames.to(device)

            outputs = model(frames)

            probs = torch.softmax(outputs, dim=1)

            all_probs.append(probs.cpu().numpy())

    avg_probs = sum(all_probs) / len(all_probs)

    predicted_class = avg_probs.argmax(axis=1)[0]

    return avg_probs, predicted_class


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model = create_model(
        num_classes=2,
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout
    )

    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"Model loaded from {args.model_path}")

    video_extensions = ['.mp4', '.avi', '.mov', '.mkv']

    if os.path.isfile(args.input):
        input_path = Path(args.input)
        if input_path.suffix.lower() in video_extensions:
            print(f"Processing video: {args.input}")
            print("Note: Please extract faces first using face_extractor.py")
            print("Provide the directory containing extracted face frames")
            return

        video_dirs = [args.input]
    else:
        video_dirs = [str(d) for d in Path(args.input).iterdir() if d.is_dir()]

    print(f"\nFound {len(video_dirs)} video directories to process")

    for video_dir in video_dirs:
        video_name = Path(video_dir).name
        print(f"\nProcessing: {video_name}")

        dataset = VideoInferenceDataset(
            video_dir=video_dir,
            num_frames=args.num_frames
        )

        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False
        )

        probs, pred_class = predict(model, dataloader, device)

        label = "FAKE" if pred_class == 1 else "REAL"
        confidence = probs[0][pred_class] * 100

        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.2f}%")
        print(f"Probabilities: Real={probs[0][0]*100:.2f}%, Fake={probs[0][1]*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Deep Fake Detection')

    parser.add_argument('--input', type=str, required=True,
                        help='Input video file or directory containing extracted faces')
    parser.add_argument('--model-path', type=str, default='models/best_model.pth',
                        help='Path to trained model checkpoint')

    parser.add_argument('--num-frames', type=int, default=40,
                        help='Number of frames per video sequence')

    parser.add_argument('--hidden-size', type=int, default=512,
                        help='LSTM hidden size (must match training)')
    parser.add_argument('--num-lstm-layers', type=int, default=2,
                        help='Number of LSTM layers (must match training)')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability (must match training)')

    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')

    args = parser.parse_args()
    main(args)