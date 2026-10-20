"""Predict a class for one patch from the demo dataset."""
from pathlib import Path
import numpy as np
import torch
from src.dataset import PatchDataset, CLASS_NAMES
from src.model import DeforestationCNN

ROOT = Path(__file__).resolve().parent

def main():
    data_path = ROOT / "data" / "demo_patches.npz"
    model_path = ROOT / "models" / "deforestation_cnn.pt"
    if not data_path.exists():
        raise SystemExit("Data belum ada. Jalankan: python generate_demo_data.py")
    if not model_path.exists():
        raise SystemExit("Model belum ada. Jalankan: python -m src.train --epochs 8")
    ds = PatchDataset(data_path)
    checkpoint = torch.load(model_path, map_location="cpu")
    model = DeforestationCNN(in_channels=checkpoint["channels"],
                             num_classes=len(checkpoint["class_names"]))
    model.load_state_dict(checkpoint["state_dict"]); model.eval()
    index = 0
    with torch.no_grad():
        probs = torch.softmax(model(torch.from_numpy(ds.X[index:index+1])), dim=1)[0].numpy()
    print(f"Demo patch index: {index}")
    print("Prediksi kelas:")
    for i in np.argsort(probs)[::-1]:
        print(f"  {CLASS_NAMES[i]:28s} {probs[i]:.3f}")
    print("Catatan: prediksi ini berasal dari patch sintetis, bukan citra satelit asli.")

if __name__ == "__main__":
    main()
