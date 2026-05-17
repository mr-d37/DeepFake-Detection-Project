# DeepFake Detection Project

A PyTorch-based deep learning project for detecting deepfake videos using ResNext50 for frame-level feature extraction and LSTM for sequence classification.

## Project Structure

```
deepfake_project/
├── data/              # Placeholder for raw video files
│   ├── real/         # Real videos go here
│   └── fake/         # Fake videos go here
├── dataset/          # Extracted face frames
│   ├── real/         # Extracted faces from real videos
│   └── fake/         # Extracted faces from fake videos
├── models/           # Saved model checkpoints
├── face_extractor.py # Face detection and extraction
├── dataset.py        # PyTorch Dataset class
├── model.py          # ResNext50 + LSTM architecture
├── train.py          # Training script
├── predict.py        # Inference script
├── requirements.txt  # Project dependencies
└── README.md         # This file
```

## Setup

1. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install dlib (required by face_recognition)**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install cmake libopenblas-dev liblapack-dev

   # Windows
   # Download and install CMake from https://cmake.org/download/

   pip install dlib
   pip install face-recognition
   ```

## Usage

### Step 1: Prepare Your Data

Place your videos in the appropriate directories:
- Real videos: `data/real/`
- Fake videos: `data/fake/`

### Step 2: Extract Faces from Videos

Extract faces from all videos in the data directory:

```bash
# Extract faces from real videos
python face_extractor.py --input-dir data/real --output-dir dataset/real --label real --num-frames 40

# Extract faces from fake videos
python face_extractor.py --input-dir data/fake --output-dir dataset/fake --label fake --num-frames 40
```

This will create directories under `dataset/real/` and `dataset/fake/`, each containing face crops from individual videos.

### Step 3: Train the Model

```bash
python train.py \
    --real-dir dataset/real \
    --fake-dir dataset/fake \
    --num-frames 40 \
    --batch-size 8 \
    --num-epochs 20 \
    --learning-rate 0.001 \
    --output-dir models
```

Training options:
- `--num-frames`: Frames per video sequence (default: 40)
- `--batch-size`: Training batch size (default: 8)
- `--num-epochs`: Number of training epochs (default: 20)
- `--learning-rate`: Initial learning rate (default: 0.001)
- `--hidden-size`: LSTM hidden size (default: 512)
- `--num-lstm-layers`: Number of LSTM layers (default: 2)
- `--dropout`: Dropout probability (default: 0.5)

The best model (based on validation accuracy) will be saved to `models/best_model.pth`.

### Step 4: Make Predictions

To classify a new video:

```bash
# First extract faces from your video
python face_extractor.py --input-dir path/to/your/video.mp4 --output-dir temp_faces --label test --num-frames 40

# Then run prediction
python predict.py --input temp_faces --model-path models/best_model.pth
```

## Model Architecture

- **Feature Extractor**: ResNext50 (pretrained on ImageNet, frozen)
  - Extracts 2048-dimensional features from each frame
- **Temporal Modeling**: LSTM
  - 2 layers with 512 hidden units
  - Processes sequence of 40 frames
- **Classifier**: Fully connected layers
  - 512 -> 256 -> 2 (real/fake)

## Requirements

- Python 3.8+
- PyTorch 1.9+
- OpenCV 4.5+
- face-recognition
- torchvision
- numpy
- pillow
- tqdm

## Hardware Recommendations

- **Training**: GPU with 8GB+ VRAM recommended
- **Inference**: GPU or CPU (slower)

## Tips for Better Performance

1. Use more training data (hundreds of videos per class)
2. Use higher resolution face crops (224x224 is default)
3. More frames per video sequence (40 is default)
4. Data augmentation (can be added to dataset.py)
5. Longer training with early stopping
