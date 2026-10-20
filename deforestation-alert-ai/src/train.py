"""Train and evaluate the CNN on a labeled NPZ patch dataset."""
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from .dataset import PatchDataset, CLASS_NAMES
from .model import DeforestationCNN

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "demo_patches.npz"))
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = PatchDataset(args.data)
    n_train = int(0.8 * len(ds))
    train_ds, test_ds = random_split(ds, [n_train, len(ds)-n_train],
                                     generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    model = DeforestationCNN(in_channels=ds.X.shape[1], num_classes=len(CLASS_NAMES)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    for epoch in range(args.epochs):
        model.train(); total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward(); optimizer.step()
            total_loss += loss.item() * len(yb)
        print(f"Epoch {epoch+1:02d}/{args.epochs} | loss={total_loss/len(train_ds):.4f}")
    model.eval(); y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            probs = torch.softmax(model(xb.to(device)), dim=1)
            y_pred.extend(probs.argmax(1).cpu().numpy().tolist())
            y_true.extend(yb.numpy().tolist())
    acc = float(accuracy_score(y_true, y_pred))
    report = classification_report(y_true, y_pred, labels=list(range(len(CLASS_NAMES))),
                                   target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    (ROOT / "models").mkdir(exist_ok=True); (ROOT / "outputs").mkdir(exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "class_names": CLASS_NAMES,
                "channels": int(ds.X.shape[1])}, ROOT / "models" / "deforestation_cnn.pt")
    with open(ROOT / "outputs" / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "classification_report": report,
                   "note": "Demo synthetic dataset; metrics are not real-world performance."},
                  f, indent=2)
    disp = ConfusionMatrixDisplay(confusion_matrix(y_true, y_pred), display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, xticks_rotation=35, colorbar=False)
    fig.tight_layout(); fig.savefig(ROOT / "outputs" / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(f"Test accuracy (synthetic demo): {acc:.3f}")
    print(f"Metrics: {ROOT / 'outputs' / 'metrics.json'}")
    print(f"Confusion matrix: {ROOT / 'outputs' / 'confusion_matrix.png'}")
    print(f"Model: {ROOT / 'models' / 'deforestation_cnn.pt'}")
    print("Reminder: validation on real, independent satellite data is required.")

if __name__ == "__main__":
    main()
