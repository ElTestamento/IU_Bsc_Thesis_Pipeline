# =====================================================================
#  Installationen, Importe, Konfiguration
# =====================================================================
!pip install -q ultralytics pingouin
from google.colab import drive
drive.mount('/content/drive')

# Hardware dieses Durchlaufs, dokumentiert als Labor für den Experimentaldurchlauf
import torch, os, platform, psutil
print("Hardware dieses Durchlaufs")
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_properties(0)
    print(f"  GPU: {gpu.name}")
    print(f"  VRAM: {gpu.total_memory / 1e9:.1f} GB")
    print(f"  CUDA: {torch.version.cuda}")
else:
    print("  GPU: keine (CPU-Betrieb)")
print(f"  CPU-Kerne: {os.cpu_count()}")
print(f"  Arbeitsspeicher: {psutil.virtual_memory().total / 1e9:.1f} GB")
print(f"  Python: {platform.python_version()} | PyTorch: {torch.__version__}")

import time
startzeit_experiment = time.time()
import os, tarfile, json, shutil, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from IPython.display import Image as IPImage, display
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from ultralytics import YOLO
import sklearn as skl
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from scipy.stats import pearsonr, spearmanr
import pingouin as pg
EPOCHS = 100
BASE = "/content/drive/MyDrive/Thesis"
SEED = 42 #wichitge, damit die Einzeltrainings vergleichbar sind
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")#Nutze Cuda weil in Colab trainiert
YOLO_DEVICE = 0 if torch.cuda.is_available() else "cpu"

# Datensätze (bleiben unberührt)
PHANTOM_TAR  = f"{BASE}/phantom_dataset.tar"
CLINICAL_TAR = f"{BASE}/clinical_dataset.tar"

# YOLO-Ordner
YOLO_TWO_PHANTOM  = f"{BASE}/YOLO_exp/two_level_model/phantom"
YOLO_TWO_CLINICAL = f"{BASE}/YOLO_exp/two_level_model/clinical"
YOLO_ONE_CLINICAL = f"{BASE}/YOLO_exp/one_level_model"

# ResNet-Ordner
RESNET_ONE = f"{BASE}/ResNet_exp/one_level_model"   # nur klinisch
RESNET_TWO = f"{BASE}/ResNet_exp/two_level_model"   # Phantom -> klinisch

# Klassen als (Supervisely-Titel, YOLO-Name), Reihenfolge = YOLO-ID
PHANTOM_CLASSES  = [("Cup", "Cup"), ("aperture", "aperture")]
CLINICAL_CLASSES = [("3 cup Polygon", "Cup"), ("1 Innenfläche Polygon", "aperture")]

# Crop-Klasse und Winkel-Tags für die Regression
CLINICAL_BOX_CLASS, CLINICAL_ANGLE_TAGS = "2 cup", ("Anteversion CT", "Inklination")
PHANTOM_BOX_CLASS,  PHANTOM_ANGLE_TAGS  = "Cup",   ("ante", "inc")
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# =====================================================================
#  Frische Ordnerstruktur bei jedem Lauf (Datensätze bleiben erhalten)
# ========================================================================
for exp in ["YOLO_exp", "ResNet_exp"]:
    shutil.rmtree(os.path.join(BASE, exp), ignore_errors=True)

for ordner in [YOLO_ONE_CLINICAL, YOLO_TWO_PHANTOM, YOLO_TWO_CLINICAL, RESNET_ONE, RESNET_TWO]:
    os.makedirs(ordner, exist_ok=True)

print("YOLO_exp und ResNet_exp frisch aufgesetzt.")


# =====================================================================
#  YOLO: Konvertierung, Training, Auswertung
# ========================================================================


def convert_supervisely_to_yolo(supervisely_tar, output_dir, classes, train_ratio=0.8):
    # classes: [(supervisely_titel, yolo_name), ...] in Reihenfolge der YOLO-IDs
    extract_dir = os.path.join(output_dir, "extracted_data")
    yolo_dir    = os.path.join(output_dir, "yolo_dataset")
    for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
        os.makedirs(os.path.join(yolo_dir, sub), exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    if not os.path.exists(os.path.join(extract_dir, "meta.json")):
        with tarfile.open(supervisely_tar) as tar:
            tar.extractall(path=extract_dir)
    dataset_dir = next((os.path.join(extract_dir, d) for d in os.listdir(extract_dir)
                        if d.startswith("dataset")), None)
    if not dataset_dir:
        return None
    img_dir = os.path.join(dataset_dir, "img")
    ann_dir = os.path.join(dataset_dir, "ann")
    title_to_id = {title: i for i, (title, _) in enumerate(classes)}
    names       = [name for _, name in classes]
    def to_yolo(ann_path, img_path):
        ann = json.load(open(ann_path))
        W, H = Image.open(img_path).size
        out = []
        for obj in ann.get("objects", []):
            title = obj.get("classTitle")
            if title not in title_to_id:
                continue
            ext = obj.get("points", {}).get("exterior", [])
            if not ext:
                continue
            xs = [p[0] for p in ext]; ys = [p[1] for p in ext]
            xc = (min(xs) + max(xs)) / 2 / W
            yc = (min(ys) + max(ys)) / 2 / H
            w  = (max(xs) - min(xs)) / W
            h  = (max(ys) - min(ys)) / H
            if all(0 <= v <= 1 for v in (xc, yc, w, h)):
                out.append(f"{title_to_id[title]} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        return out
    pairs = [(os.path.join(img_dir, f), os.path.join(ann_dir, f + ".json"))
             for f in os.listdir(img_dir)
             if f.endswith((".png", ".jpg", ".jpeg"))
             and os.path.exists(os.path.join(ann_dir, f + ".json"))]
    if not pairs:
        return None
    random.shuffle(pairs)
    n_train = int(len(pairs) * train_ratio)
    for idx, (img_path, ann_path) in enumerate(tqdm(pairs)):
        subset = "train" if idx < n_train else "val"
        fn = os.path.basename(img_path)
        shutil.copy2(img_path, os.path.join(yolo_dir, "images", subset, fn))
        with open(os.path.join(yolo_dir, "labels", subset, os.path.splitext(fn)[0] + ".txt"), "w") as f:
            f.write("\n".join(to_yolo(ann_path, img_path)))
    with open(os.path.join(yolo_dir, "classes.txt"), "w") as f:
        f.write("\n".join(names) + "\n")
    yaml_path = os.path.join(yolo_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(f"path: {yolo_dir}\ntrain: images/train\nval: images/val\n"
                f"nc: {len(names)}\nnames: {names}\n")
    return yaml_path


def train_yolo(yaml_path, output_dir, run_name, weights, lr0=0.01, epochs=EPOCHS, batch=16):
    # weights='yolov8m.pt' -> weights=<best.pt> -> Fine-Tuning
    model = YOLO(weights)
    model.train(data=yaml_path, epochs=epochs, imgsz=640, batch=batch, patience=20,
                lr0=lr0, project=output_dir, name=run_name, exist_ok=True, device=YOLO_DEVICE)
    return os.path.join(output_dir, run_name, "weights", "best.pt")


def evaluate_yolo(run_dir, titel):
    print("\n" + "=" * 70)
    print(f"YOLO-Auswertung: {titel}")
    print("=" * 70)
    csv = os.path.join(run_dir, "results.csv")
    if not os.path.exists(csv):
        print("results.csv nicht gefunden:", csv)
        return
    df = pd.read_csv(csv)
    df.columns = df.columns.str.strip()
    def col(sub):
        hits = [c for c in df.columns if sub in c]
        return hits[0] if hits else None
    # Bester Epoch nach mAP50 (nicht die letzte Epoche, wegen Early Stopping)
    c_map = col("mAP50(B)")
    best = df.loc[df[c_map].idxmax()] if c_map else df.iloc[-1]
    for label, sub in [("mAP50", "mAP50(B)"), ("Precision", "precision(B)"), ("Recall", "recall(B)")]:
        c = col(sub)
        if c:
            print(f"  {label:10s}: {best[c]:.4f}")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for c in df.columns:
        if "train" in c and "loss" in c:
            ax[0].plot(df["epoch"], df[c], label=c)
        if "val" in c and "loss" in c:
            ax[0].plot(df["epoch"], df[c], "--", label=c)
    ax[0].set_title(f"{titel}: Loss"); ax[0].set_xlabel("Epoche"); ax[0].set_ylabel("Loss")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=0.3)
    if c_map:
        ax[1].plot(df["epoch"], df[c_map])
        ax[1].set_title(f"{titel}: mAP50"); ax[1].set_xlabel("Epoche"); ax[1].set_ylabel("mAP50")
        ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    example = os.path.join(run_dir, "val_batch0_pred.jpg")#BSP-Bildraster kommt von YOLO!
    if os.path.exists(example):
        print("Detektionsbeispiel (Val):")
        display(IPImage(example))


#  =====================================================================
#  YOLO-Durchlaeufe (Subfrage A: Transfer A1 vs. Baseline A2)
# ========================================================================
# A1 Phantom-Vortraining (Stufe 1, ab COCO)
startzeit_YOLO_transfertraining = time.time()
print()
print("#################### Daten werden aufbereitet ######################################")
print("####################################################################################")
yaml_phantom = convert_supervisely_to_yolo(PHANTOM_TAR, YOLO_TWO_PHANTOM, PHANTOM_CLASSES)
print()
print("#################### YOLO-Phantomtraining startet ######################################")
print("########################################################################################")
phantom_best = train_yolo(yaml_phantom, YOLO_TWO_PHANTOM, "phantom_pretrain",
                          weights="yolov8m.pt", lr0=0.01, batch=16)

evaluate_yolo(os.path.join(YOLO_TWO_PHANTOM, "phantom_pretrain"), "A1: Phantom-Vortraining (Stufe 1)")
print()
print("#################### YOLO-Fine-Tuning auf klinischen Daten ######################################")
print("################################################################################################")

# Klinische Daten nur einmal konvertieren, damit A1 und A2 exakt denselben Split sehen.
# Seed direkt davor neu setzen, damit der klinische Split unabhängig vom Phantomlauf reproduzierbar ist.
random.seed(SEED)
yaml_clinical = convert_supervisely_to_yolo(CLINICAL_TAR, YOLO_TWO_CLINICAL, CLINICAL_CLASSES)

# A2 Transfermodell: klinisches Fine-Tuning ab Phantom-Gewichten
transfer_best = train_yolo(yaml_clinical, YOLO_TWO_CLINICAL, "clinical_finetune",
                           weights=phantom_best, lr0=0.01, batch=32)

evaluate_yolo(os.path.join(YOLO_TWO_CLINICAL, "clinical_finetune"), "A2: Transfermodell (Phantom -> klinisch)")
endzeit_YOLO_transfertraining = time.time()
print()
print("#################### YOLO-Training nur auf klinischen Daten ohne Vortraining (Phantom) ######################################")
print("#############################################################################################################################")

# A3 Einstufen-Modell: gleicher Split und gleiche Lernrate wie A2, einziger Unterschied ist das fehlende Phantom-Vortraining
startzeit_YOLO_kein_transfer = time.time()
baseline_best = train_yolo(yaml_clinical, YOLO_ONE_CLINICAL, "clinical_baseline",
                           weights="yolov8m.pt", lr0=0.01, batch=32)

evaluate_yolo(os.path.join(YOLO_ONE_CLINICAL, "clinical_baseline"), "A3: Einstufen-Modell (nur klinisch)")
endzeit_YOLO_kein_transfer = time.time()


# =====================================================================
#  Regression: Mit einem Netz und mit zwei Netzen mit getrennten Köpfen pro Winkel
# =========================================================================
# Cave: Keine geometrieverändernde Augmentierung. Flip/Rotation würden die Winkel-Labels verfälschen.
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

])
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),

])


class My_Dataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB").crop(row["box_coords"])
        if self.transform:
            image = self.transform(image)
        labels = torch.tensor([row["anteversion"], row["inklination"]], dtype=torch.float32)
        return image, labels


def extract_dataset(tar_path, extract_root):
    os.makedirs(extract_root, exist_ok=True)
    if not any(d.startswith("dataset") for d in os.listdir(extract_root)):
        with tarfile.open(tar_path) as tar:
            tar.extractall(path=extract_root)
    dataset_dir = next(os.path.join(extract_root, d) for d in os.listdir(extract_root) if d.startswith("dataset"))
    return os.path.join(dataset_dir, "ann"), os.path.join(dataset_dir, "img")


def build_dataframe(ann_dir, img_dir, box_class, ante_tag, inkl_tag, pad=1.0):
    # Crop-Box aus den Außenpunkten von box_class (funktioniert für Bounding Box und Polygon)
    rows = []
    for file in os.listdir(ann_dir):
        with open(os.path.join(ann_dir, file)) as f:
            ann = json.load(f)
        tags = {t["name"]: t["value"] for t in ann.get("tags", [])}
        box = None
        for obj in ann.get("objects", []):
            if obj["classTitle"] == box_class:
                xs = [p[0] for p in obj["points"]["exterior"]]
                ys = [p[1] for p in obj["points"]["exterior"]]
                cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
                w, h = (max(xs) - min(xs)) * pad, (max(ys) - min(ys)) * pad
                box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                break
        rows.append({
            "image_path":  os.path.join(img_dir, file[:-5]),
            "anteversion": tags.get(ante_tag),
            "inklination": tags.get(inkl_tag),
            "box_coords":  box,
        })
    return pd.DataFrame(rows).dropna().reset_index(drop=True)


def make_loaders(df, batch_size=32):
    # Winkel z-normieren, Statistik für die Rückrechnung zurückgeben, dann 80/20 splitten
    stats = {
        "ante_mean": df["anteversion"].mean(), "ante_sd": df["anteversion"].std(),
        "inkl_mean": df["inklination"].mean(), "inkl_sd": df["inklination"].std(),
    }
    df = df.copy()
    df["anteversion"] = (df["anteversion"] - stats["ante_mean"]) / stats["ante_sd"]
    df["inklination"] = (df["inklination"] - stats["inkl_mean"]) / stats["inkl_sd"]
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=SEED)
    tr = DataLoader(My_Dataset(df_train, train_transform), batch_size=batch_size, shuffle=True)
    va = DataLoader(My_Dataset(df_val,   val_transform),   batch_size=batch_size, shuffle=False)
    return tr, va, stats


def compute_icc(true_vals, pred_vals, name):
    n = len(true_vals)
    data = pd.DataFrame({
        "subject": list(range(n)) * 2,
        "rater":   ["CT"] * n + ["Model"] * n,
        "value":   list(true_vals) + list(pred_vals),
    })
    res = pg.intraclass_corr(data=data, targets="subject", raters="rater", ratings="value")
    icc2 = res[res["Type"] == "ICC(A,1)"].iloc[0]
    lo, hi = icc2["CI95"]
    print(f"  {name}: ICC(A,1) = {icc2['ICC']:.3f}  (95% CI {lo:.2f}-{hi:.2f})")


def bland_altman(pred, true, titel):
    mean = (pred + true) / 2
    diff = pred - true
    bias, sd = np.mean(diff), np.std(diff)
    plt.figure(figsize=(7, 5))
    plt.scatter(mean, diff, alpha=0.6)
    plt.axhline(bias, color="red", label=f"Bias: {bias:.2f}°")
    plt.axhline(bias + 1.96 * sd, color="gray", ls="--", label=f"+1.96 SD: {bias + 1.96*sd:.2f}°")
    plt.axhline(bias - 1.96 * sd, color="gray", ls="--", label=f"-1.96 SD: {bias - 1.96*sd:.2f}°")
    plt.xlabel("Mittelwert (Modell + GT) / 2  [°]")
    plt.ylabel("Differenz (Modell − GT)  [°]")
    plt.title(f"Bland-Altman: {titel}"); plt.legend(); plt.tight_layout(); plt.show()


def metrics_block(pred, true, mean_train, name):
    err = np.abs(pred - true)
    boot = []
    for _ in range(1000):
        idx = resample(range(len(true)), n_samples=len(true))
        boot.append(np.abs(pred[idx] - true[idx]).mean())
    ci = np.percentile(boot, [2.5, 97.5])
    print(f"\n{name}")
    print(f"  MAE {err.mean():.2f}° (95% CI {ci[0]:.2f}-{ci[1]:.2f}) | Median {np.median(err):.2f}° | Baseline {np.abs(true - mean_train).mean():.2f}°")
    print(f"  RMSE {skl.metrics.root_mean_squared_error(true, pred):.2f}° | R² {skl.metrics.r2_score(true, pred):.2f}")
    print(f"  Pearson {pearsonr(true, pred)[0]:.3f} | Spearman {spearmanr(true, pred)[0]:.3f}")
    print(f"  ±5° {(err <= 5).mean()*100:.1f}% | ±10° {(err <= 10).mean()*100:.1f}%")


def scatter_and_bland(pred_ante, true_ante, pred_inkl, true_inkl, titel):
    # gemeinsame Plots für beide Winkel
    for pred, true, name in [(pred_ante, true_ante, "Anteversion"), (pred_inkl, true_inkl, "Inklination")]:
        plt.figure(figsize=(6, 5))
        sns.scatterplot(x=true, y=pred)
        sns.regplot(x=true, y=pred, scatter=False, color="green")
        plt.xlabel(f"Ground Truth {name}"); plt.ylabel(f"Vorhersage {name}")
        plt.title(f"{titel} — {name}"); plt.tight_layout(); plt.show()
    bland_altman(pred_ante, true_ante, f"{titel} — Anteversion")
    bland_altman(pred_inkl, true_inkl, f"{titel} — Inklination")


def report_metrics(pred_ante, true_ante, pred_inkl, true_inkl, stats):
    metrics_block(pred_ante, true_ante, stats["ante_mean"], "Anteversion")
    metrics_block(pred_inkl, true_inkl, stats["inkl_mean"], "Inklination")
    print("\nICC(A,1):")
    compute_icc(true_ante, pred_ante, "Anteversion")
    compute_icc(true_inkl, pred_inkl, "Inklination")


def denorm(values, mean, sd):
    return np.array(values) * sd + mean


# ========================================================================
#  Regression Multi-Task: ein Backbone, zwei Köpfe
# =========================================================================


class My_neural_layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model_backbone.fc = nn.Identity()
        self.anteversion_head_layer = nn.Linear(512, 1)
        self.inklination_head_layer = nn.Linear(512, 1)
    def forward(self, x):
        feats = self.model_backbone(x)
        return self.anteversion_head_layer(feats), self.inklination_head_layer(feats)


def train_regression(train_loader, val_loader, save_path, epochs=EPOCHS, lr=2e-4, init_weights=None):
    # init_weights gesetzt -> von diesen Gewichten weitertrainieren (Transfer)
    model = My_neural_layer().to(device)
    if init_weights:
        model.load_state_dict(torch.load(init_weights, map_location=device))
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_epoch = float("inf"), -1
    hist_train, hist_val = [], []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for image, labels in train_loader:
            image = image.to(device)
            y_ante = labels[:, 0].to(device).unsqueeze(1)
            y_inkl = labels[:, 1].to(device).unsqueeze(1)
            optimizer.zero_grad()
            p_ante, p_inkl = model(image)
            loss = loss_fn(p_ante, y_ante) + loss_fn(p_inkl, y_inkl)
            loss.backward(); optimizer.step()
            running += loss.item()
        avg = running / len(train_loader)
        model.eval()
        running_v = 0.0
        with torch.no_grad():
            for image, labels in val_loader:
                image = image.to(device)
                y_ante = labels[:, 0].to(device).unsqueeze(1)
                y_inkl = labels[:, 1].to(device).unsqueeze(1)
                p_ante, p_inkl = model(image)
                running_v += (loss_fn(p_ante, y_ante) + loss_fn(p_inkl, y_inkl)).item()
        avg_v = running_v / len(val_loader)
        hist_train.append(avg); hist_val.append(avg_v)
        print(f"Epoch {epoch+1:>2}/{epochs} | Train {avg:.4f} | Val {avg_v:.4f}")
        if avg_v < best_val:
            best_val, best_epoch = avg_v, epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"   -> bester Stand (Val {best_val:.4f}, Epoche {best_epoch})")
    return {"train": hist_train, "val": hist_val, "best_val": best_val, "best_epoch": best_epoch}


def evaluate_regression(model_path, val_loader, stats, hist, titel):
    print("\n" + "=" * 70)
    print(f"Regression-Auswertung: {titel}")
    print("=" * 70)
    plt.figure(figsize=(8, 5))
    sns.set_style("darkgrid")
    sns.lineplot(data=pd.DataFrame({"Train Loss": hist["train"], "Val Loss": hist["val"]}))
    plt.title(f"{titel}: Train/Val Loss"); plt.xlabel("Epoche"); plt.ylabel("Loss"); plt.show()
    model = My_neural_layer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    pa, pi, ta, ti = [], [], [], []
    with torch.no_grad():
        for image, labels in val_loader:
            image = image.to(device)
            out_a, out_i = model(image)
            pa.append(out_a.cpu().reshape(-1)); pi.append(out_i.cpu().reshape(-1))
            ta.append(labels[:, 0]);            ti.append(labels[:, 1])
    pred_ante = denorm(torch.cat(pa), stats["ante_mean"], stats["ante_sd"])
    true_ante = denorm(torch.cat(ta), stats["ante_mean"], stats["ante_sd"])
    pred_inkl = denorm(torch.cat(pi), stats["inkl_mean"], stats["inkl_sd"])
    true_inkl = denorm(torch.cat(ti), stats["inkl_mean"], stats["inkl_sd"])
    report_metrics(pred_ante, true_ante, pred_inkl, true_inkl, stats)
    scatter_and_bland(pred_ante, true_ante, pred_inkl, true_inkl, titel)


# ==========================================================================
#  Regression Einzelregressoren: zwei getrennte Modelle, je ein Kopf (pro Winkelart)
# ========================================================================


class My_neural_layer_single_head(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model_backbone.fc = nn.Identity()
        self.angle_layer = nn.Linear(512, 1)   # ein Ausgang, ein Winkel
    def forward(self, x):
        feats = self.model_backbone(x)
        return self.angle_layer(feats)


def train_one_regressor(train_loader, val_loader, ziel, save_path, epochs=EPOCHS, lr=2e-4):
    # ziel = 0 -> Anteversion, ziel = 1 -> Inklination
    model = My_neural_layer_single_head().to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val, best_epoch = float("inf"), -1
    hist_train, hist_val = [], []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for image, labels in train_loader:
            image = image.to(device)
            y = labels[:, ziel].to(device).unsqueeze(1)
            optimizer.zero_grad()
            p = model(image)
            loss = loss_fn(p, y)
            loss.backward(); optimizer.step()
            running += loss.item()
        avg = running / len(train_loader)
        model.eval()
        running_v = 0.0
        with torch.no_grad():
            for image, labels in val_loader:
                image = image.to(device)
                y = labels[:, ziel].to(device).unsqueeze(1)
                p = model(image)
                running_v += loss_fn(p, y).item()
        avg_v = running_v / len(val_loader)
        hist_train.append(avg); hist_val.append(avg_v)
        print(f"Epoch {epoch+1:>2}/{epochs} | Train {avg:.4f} | Val {avg_v:.4f}")
        if avg_v < best_val:
            best_val, best_epoch = avg_v, epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"   -> bester Stand (Val {best_val:.4f}, Epoche {best_epoch})")
    return {"train": hist_train, "val": hist_val, "best_val": best_val, "best_epoch": best_epoch}


def train_single_regressors(train_loader, val_loader, save_path_ante, save_path_inkl, epochs=EPOCHS, lr=2e-4):
    # Zwei unabhängige Einzelmodelle
    print("--- Einzelregressor Anteversion ---")
    hist_ante = train_one_regressor(train_loader, val_loader, 0, save_path_ante, epochs, lr)
    print("--- Einzelregressor Inklination ---")
    hist_inkl = train_one_regressor(train_loader, val_loader, 1, save_path_inkl, epochs, lr)
    return {"ante": hist_ante, "inkl": hist_inkl}


def evaluate_single_regressors(ante_path, inkl_path, val_loader, stats, hist, titel):
    print("\n" + "=" * 70)
    print(f"Regression-Auswertung: {titel}")
    print("=" * 70)
    # Loss-Kurven beider Einzelmodelle
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(hist["ante"]["train"], label="Train"); ax[0].plot(hist["ante"]["val"], label="Val")
    ax[0].set_title("Anteversion: Loss"); ax[0].set_xlabel("Epoche"); ax[0].set_ylabel("Loss")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(hist["inkl"]["train"], label="Train"); ax[1].plot(hist["inkl"]["val"], label="Val")
    ax[1].set_title("Inklination: Loss"); ax[1].set_xlabel("Epoche"); ax[1].set_ylabel("Loss")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    plt.tight_layout(); plt.show()
    model_ante = My_neural_layer_single_head().to(device)
    model_ante.load_state_dict(torch.load(ante_path, map_location=device))
    model_ante.eval()
    model_inkl = My_neural_layer_single_head().to(device)
    model_inkl.load_state_dict(torch.load(inkl_path, map_location=device))
    model_inkl.eval()
    pa, pi, ta, ti = [], [], [], []
    with torch.no_grad():
        for image, labels in val_loader:
            image = image.to(device)
            pa.append(model_ante(image).cpu().reshape(-1))
            pi.append(model_inkl(image).cpu().reshape(-1))
            ta.append(labels[:, 0]); ti.append(labels[:, 1])
    pred_ante = denorm(torch.cat(pa), stats["ante_mean"], stats["ante_sd"])
    true_ante = denorm(torch.cat(ta), stats["ante_mean"], stats["ante_sd"])
    pred_inkl = denorm(torch.cat(pi), stats["inkl_mean"], stats["inkl_sd"])
    true_inkl = denorm(torch.cat(ti), stats["inkl_mean"], stats["inkl_sd"])
    report_metrics(pred_ante, true_ante, pred_inkl, true_inkl, stats)
    scatter_and_bland(pred_ante, true_ante, pred_inkl, true_inkl, titel)


# ========================================================================
#  Regression-Durchläufe
# ========================================================================
# Klinische Daten einmal vorbereiten. Alle Bedingungen nutzen denselben Split.
print()
print("#################### Datenvorbereitung für Regressionstrainings ######################################")
print("######################################################################################################")
startzeit_prepare_dataset = time.time()
ann_c, img_c = extract_dataset(CLINICAL_TAR, f"{BASE}/ResNet_exp/clinical_extracted")
df_clin = build_dataframe(ann_c, img_c, CLINICAL_BOX_CLASS, *CLINICAL_ANGLE_TAGS)
tr_clin, va_clin, stats_clin = make_loaders(df_clin)
endzeit_prepare_dataset = time.time()

# Phantomdaten für das Vortraining der Multi-Task-Transfer-Bedingung
print()
print("#################### Regressionstraining der Phantomdaten ############################################")
print("######################################################################################################")
startzeit_regression_transfer = time.time()
ann_p, img_p = extract_dataset(PHANTOM_TAR, f"{BASE}/ResNet_exp/phantom_extracted")
df_phan = build_dataframe(ann_p, img_p, PHANTOM_BOX_CLASS, *PHANTOM_ANGLE_TAGS)
tr_phan, va_phan, _ = make_loaders(df_phan)

# Block B (Transfer, Multi-Task): B1 nur klinisch vs. B2 Phantom und dann klinisch
print()
print("#################### Regressionstraining (Zwei Köpfe) nur auf den klinischen Daten ############################################")
print("###############################################################################################################################")
hist_multi_b1 = train_regression(tr_clin, va_clin, f"{RESNET_ONE}/best_model.pt")
evaluate_regression(f"{RESNET_ONE}/best_model.pt", va_clin, stats_clin, hist_multi_b1,
                    "B1: Multi-Task, nur klinisch")

print()
print("#################### Regressionstraining (Zwei Köpfe): Transfertraining erst auf Phantomdaten, dann Fine-Tuning auf den klinischen Daten ####################################")
print("###############################################################################################################################################################")
train_regression(tr_phan, va_phan, f"{RESNET_TWO}/phantom_pretrained.pt")
hist_multi_b2 = train_regression(tr_clin, va_clin, f"{RESNET_TWO}/best_model.pt",
                                 init_weights=f"{RESNET_TWO}/phantom_pretrained.pt")

evaluate_regression(f"{RESNET_TWO}/best_model.pt", va_clin, stats_clin, hist_multi_b2,
                    "B2: Multi-Task, zweistufig (Phantom -> klinisch)")

endzeit_regression_transfer = time.time()

# Block C (Architektur, nur klinisch): Multi-Task (B1) vs. zwei Einzelregressoren (B3)
print()
print("#################### Regressionstraining (Ein Kopf pro Winkel) nur auf den klinischen Daten ############################################")
print("########################################################################################################################################")
startzeit_regression_single_head = time.time()
hist_single = train_single_regressors(tr_clin, va_clin,
                                      f"{RESNET_ONE}/best_ante_model.pt",
                                      f"{RESNET_ONE}/best_inkl_model.pt")

endzeit_regression_single_head = time.time()
evaluate_single_regressors(f"{RESNET_ONE}/best_ante_model.pt", f"{RESNET_ONE}/best_inkl_model.pt",
                           va_clin, stats_clin, hist_single, "B3: Einzelregressoren, nur klinisch")

endzeit_experiment = time.time()

#Gemessene Zeiten:
print()
print(f"Dauer der Aufbereitung der exportierten Daten aus Supervisely: {(endzeit_prepare_dataset - startzeit_prepare_dataset)/60:.1f} Minuten")
print(f"Das gesamte KI-Experimente dauert: {(endzeit_experiment - startzeit_experiment)/60:.1f} Minuten")
print(f"Dauer des YOLO-Transferlernens : {(endzeit_YOLO_transfertraining-startzeit_YOLO_transfertraining)/60:.1f} Minuten")
print(f"Dauer des YOLO-kein-Transferlernens : {(endzeit_YOLO_kein_transfer-startzeit_YOLO_kein_transfer)/60:.1f} Minuten")
print(f"Dauer des Regression-Transferlernens : {(endzeit_regression_transfer-startzeit_regression_transfer)/60:.1f} Minuten")
print(f"Dauer des Regressionlernens für zwei separate Köpfe: {(endzeit_regression_single_head-startzeit_regression_single_head)/60:.1f} Minuten")

import sys, platform
print(sys.version)
print(platform.python_version())   # nur "3.11.13"

import glob, os, pandas as pd, matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/drive')


BASE = "/content/drive/MyDrive/CV3"
csvs = sorted(glob.glob(f"{BASE}/**/results.csv", recursive=True))

# 1) Endmetriken je Lauf -> so erkennst du den definitiven Lauf
print("Lauf".ljust(52), "mAP50   mAP50-95")
for f in csvs:
    d = pd.read_csv(f); d.columns = d.columns.str.strip()
    c50 = [c for c in d.columns if "mAP50(B)" in c or c.endswith("mAP50")]
    c95 = [c for c in d.columns if "mAP50-95(B)" in c or c.endswith("mAP50-95")]
    last = d.iloc[-1]
    v50 = float(last[c50[0]]) if c50 else float("nan")
    v95 = float(last[c95[0]]) if c95 else float("nan")
    print(f.replace(BASE + "/", "").ljust(52), f"{v50:.3f}   {v95:.3f}")

# 2) robuste Plot-Funktion (Spike wird über den eingeschwungenen Bereich ignoriert)
import math, pandas as pd, matplotlib.pyplot as plt

def plot_yolo_loss(csv_path, titel, out_png, settle_from=25):
    d = pd.read_csv(csv_path); d.columns = d.columns.str.strip()
    if "epoch" not in d.columns:
        d["epoch"] = range(1, len(d) + 1)
    loss_cols = [c for c in d.columns if c.endswith("loss")]
    d[loss_cols] = d[loss_cols].apply(pd.to_numeric, errors="coerce")
    ymax = d.loc[d["epoch"] >= settle_from, loss_cols].max().max()   # pandas ignoriert NaN
    if not (isinstance(ymax, float) and math.isfinite(ymax)) or ymax <= 0:
        ymax = d[loss_cols].max().max()
    ymax = float(ymax) * 2.0 if math.isfinite(float(ymax)) and ymax > 0 else 3.0
    ax = d.plot(x="epoch", y=loss_cols, figsize=(7, 4), linewidth=1.2)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Epoche"); ax.set_ylabel("Loss"); ax.set_title(titel)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(out_png, dpi=200, bbox_inches="tight"); plt.show()

# 3) alle Läufe plotten (Ordnerpfad als Titel)
for f in csvs:
    label = f.replace(BASE + "/", "").replace("/results.csv", "")
    plot_yolo_loss(f, label, label.replace("/", "_") + "_loss.png")

!find /content/drive/MyDrive -maxdepth 5 -name results.csv 2>/dev/null
!find /content/runs -maxdepth 5 -name results.csv 2>/dev/null
!find runs -maxdepth 5 -name results.csv 2>/dev/null

import math, pandas as pd, matplotlib.pyplot as plt

def plot_yolo_loss(csv_path, titel, out_png, settle_from=25):
    d = pd.read_csv(csv_path); d.columns = d.columns.str.strip()
    if "epoch" not in d.columns:
        d["epoch"] = range(1, len(d) + 1)
    lc = [c for c in d.columns if c.endswith("loss")]
    d[lc] = d[lc].apply(pd.to_numeric, errors="coerce")
    ymax = d.loc[d["epoch"] >= settle_from, lc].max().max()      # Spike wird ignoriert
    if not (isinstance(ymax, float) and math.isfinite(ymax)) or ymax <= 0:
        ymax = d[lc].max().max()
    ymax = float(ymax) * 2.0 if math.isfinite(float(ymax)) and ymax > 0 else 3.0
    ax = d.plot(x="epoch", y=lc, figsize=(7, 4), linewidth=1.2)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("Epoche"); ax.set_ylabel("Loss"); ax.set_title(titel)
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout(); plt.savefig(out_png, dpi=200, bbox_inches="tight"); plt.show()

plot_yolo_loss("/content/drive/MyDrive/CV3/first_level/hip_detection/results.csv",
               "Phantom-Vortraining: Loss", "phantom_loss.png")