import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision import models
from torchsummary import summary
from torchview import draw_graph
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import cv2
from torchvision.models import vit_b_16, ViT_B_16_Weights  # Vit import

# Dataset absolute paths
base_dirs = [
    r"D:\\Major Project\\Rishi_Raj\\Image_detection\\Dataset-01\\train-20250112T065955Z-001",
    r"D:\\Major Project\\Rishi_Raj\\Image_detection\\Dataset-01\\test-20250112T065939Z-001"
]
labels = ["fake", "real"]
g_train, g_test = {"fake": [], "real": []}, {"fake": [], "real": []}
valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')

# Load image file paths
for base_dir in base_dirs:
    for label in labels:
        img_dir = os.path.join(base_dir, label)
        if not os.path.exists(img_dir):
            print(f"Folder not found: {img_dir}")
            continue
        files = [os.path.join(img_dir, x)
                 for x in os.listdir(img_dir)
                 if x.lower().endswith(valid_exts) and os.path.isfile(os.path.join(img_dir, x))]
        print(f"Found {len(files)} images in {img_dir}")
        if "train" in base_dir:
            g_train[label] += files
        else:
            g_test[label] += files

# Merge and shuffle
all_images, all_labels = [], []
for label in labels:
    all_images += g_train[label] + g_test[label]
    all_labels += [label]*len(g_train[label]) + [label]*len(g_test[label])
data = pd.DataFrame({"images": all_images, "labels": all_labels})
data = data.sample(frac=1, random_state=42).reset_index(drop=True)
label_map = {'fake': 0, 'real': 1}
data['labels'] = data['labels'].map(label_map)

# Load and verify images
Images, Labels = [], []
max_height, min_height = float('-inf'), float('inf')
max_width, min_width = float('-inf'), float('inf')
for i in range(len(data)):
    image_path, label = data['images'][i], data['labels'][i]
    img = cv2.imread(image_path)
    if img is None:
        print("Image not read or invalid:", image_path)
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    Images.append(img)
    Labels.append(label)
    h, w = img.shape[:2]
    max_height, min_height = max(max_height, h), min(min_height, h)
    max_width, min_width = max(max_width, w), min(min_width, w)
print(f"Max Height: {max_height}")
print(f"Min Height: {min_height}")
print(f"Max Width: {max_width}")
print(f"Min Width: {min_width}")

# Resize images
target_size = (224, 224)
Images_resized = [cv2.resize(img, target_size) for img in Images]

# Visualization
num_imgs = min(20, len(Images_resized))
ncols = 4
nrows = (num_imgs + ncols - 1) // ncols
plt.figure(figsize=(15, 10))
for i, img in enumerate(Images_resized[:num_imgs]):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(img)
    plt.title(f"Label: {Labels[i]}")
    plt.axis('off')
plt.show()

# Split
X_train, X_test, y_train, y_test = train_test_split(Images_resized, Labels, test_size=0.05, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=42)
print("Training data shape:", len(X_train))
print("Validation data shape:", len(X_val))
print("Testing data shape:", len(X_test))

sns.countplot(x=np.array(y_train))
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Distribution of Classes in Training Set")
plt.show()

# Transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])

# Dataset class
class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = np.array(images)
        self.labels = torch.tensor(np.array(labels), dtype=torch.long)
        self.transform = transform
    def __len__(self): return len(self.images)
    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform: image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Dataloaders
train_dataset = CustomImageDataset(X_train, y_train, transform=transform)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataset = CustomImageDataset(X_val, y_val, transform=transform)
val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
test_dataset = CustomImageDataset(X_test, y_test, transform=test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

images, labels = next(iter(train_dataloader))
print(images.shape)
print(labels)

# ==== ViT Model Block ====
model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False  # Freeze all layers
in_features = model.heads.head.in_features
model.heads.head = nn.Sequential(
    nn.Linear(in_features, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(),
    nn.Dropout(0.5),
    nn.Linear(256, 1),
    nn.Sigmoid()
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

def binary_accuracy(preds, labels):
    preds = (preds >= 0.5).float()
    return (preds == labels).sum().item() / labels.size(0)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=10, save_path="Image_detection/best_vit.pth"):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss, train_acc = 0.0, 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_acc += binary_accuracy(outputs, labels)
        model.eval()
        val_loss, val_acc = 0.0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += binary_accuracy(outputs, labels)
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Saving the Model with loss {best_val_loss:.4f}")
            torch.save(model.state_dict(), save_path)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Train Loss: {train_loss/len(train_loader):.4f}, Train Acc: {train_acc/len(train_loader):.4f} | "
              f"Val Loss: {val_loss/len(val_loader):.4f}, Val Acc: {val_acc/len(val_loader):.4f}| LR:{current_lr:.6f} ")

# Train
train(model, train_dataloader, val_dataloader, criterion, optimizer, scheduler, epochs=10, save_path="Image_detection/best_vit.pth")

# Test
model.eval()
y_true, y_pred_probs = [], []
with torch.no_grad():
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device).float().unsqueeze(1)
        outputs = model(images)
        y_true.extend(labels.cpu().numpy().flatten().tolist())
        y_pred_probs.extend(outputs.cpu().numpy().flatten().tolist())
y_pred = [1 if p >= 0.5 else 0 for p in y_pred_probs]
test_acc = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {test_acc:.4f}")

conf_mat = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", conf_mat)
class_report = classification_report(y_true, y_pred, target_names=['fake', 'real'])
print("Classification Report:\n", class_report)

# Visualize confusion matrix as image
plt.figure(figsize=(6, 5))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=['fake', 'real'], yticklabels=['fake', 'real'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualize predictions
def plot_test_images(image_list, labels, preds, class_names, num_images=10):
    num_images = min(num_images, len(image_list))
    ncols = 5
    nrows = int(np.ceil(num_images / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 8))
    axes = axes.flatten()
    for i in range(num_images):
        img = image_list[i]
        actual_label = class_names[labels[i]]
        pred_label = class_names[preds[i]]
        axes[i].imshow(img)
        axes[i].set_title(f"Actual: {actual_label}\nPred: {pred_label}", fontsize=10)
        axes[i].axis("off")
    for i in range(num_images, len(axes)):
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()
plot_test_images(image_list=X_test, labels=y_true, preds=y_pred, class_names={0: "fake", 1: "real"}, num_images=len(X_test))

# Model visualization (torchview), with only supported args
draw_graph(model, input_size=(1, 3, 224, 224), show_shapes=True)