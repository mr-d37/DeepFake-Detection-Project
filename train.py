import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from pathlib import Path

from dataset import DeepFakeDataset
from model import create_model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.

    Args:
        model: The neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu)

    Returns:
        Average training loss and accuracy for the epoch
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for frames, labels in tqdm(dataloader, desc="Training"):
        frames = frames.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(frames)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """
    Validate the model on validation set.

    Args:
        model: The neural network model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to validate on (cuda/cpu)

    Returns:
        Average validation loss and accuracy
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for frames, labels in tqdm(dataloader, desc="Validating"):
            frames = frames.to(device)
            labels = labels.to(device)

            outputs = model(frames)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100.0 * correct / total

    return epoch_loss, epoch_acc


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")

    if args.data_dir:
        real_dir = os.path.join(args.data_dir, 'real')
        fake_dir = os.path.join(args.data_dir, 'fake')
    else:
        real_dir = args.real_dir
        fake_dir = args.fake_dir

    print("Loading dataset...")
    dataset = DeepFakeDataset(
        real_dir=real_dir,
        fake_dir=fake_dir,
        num_frames=args.num_frames
    )

    print(f"Total samples: {len(dataset)}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    print("Creating model...")
    model = create_model(
        num_classes=2,
        hidden_size=args.hidden_size,
        num_lstm_layers=args.num_lstm_layers,
        dropout=args.dropout
    )
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

    os.makedirs(args.output_dir, exist_ok=True)
    best_val_acc = 0.0

    print("\nStarting training...")
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        print("-" * 50)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_acc = validate(
            model, val_loader, criterion, device
        )

        scheduler.step()

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(args.output_dir, 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, checkpoint_path)
            print(f"Best model saved with validation accuracy: {val_acc:.2f}%")

    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved to: {args.output_dir}/best_model.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Deep Fake Detection Model')

    parser.add_argument('--real-dir', type=str, default='dataset/real',
                        help='Directory containing real video face extractions')
    parser.add_argument('--fake-dir', type=str, default='dataset/fake',
                        help='Directory containing fake video face extractions')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Parent directory containing real/ and fake/ subdirectories')

    parser.add_argument('--num-frames', type=int, default=40,
                        help='Number of frames per video sequence')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size for training')
    parser.add_argument('--num-epochs', type=int, default=20,
                        help='Number of training epochs')

    parser.add_argument('--hidden-size', type=int, default=512,
                        help='LSTM hidden size')
    parser.add_argument('--num-lstm-layers', type=int, default=2,
                        help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability')

    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='Weight decay for optimizer')

    parser.add_argument('--output-dir', type=str, default='models',
                        help='Directory to save model checkpoints')
    parser.add_argument('--cpu', action='store_true',
                        help='Force CPU usage even if CUDA is available')

    args = parser.parse_args()
    main(args)