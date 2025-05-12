# train.py

import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split

from model.layers import ModeloCNN

# 1) Defina aqui as classes EXATAMENTE como são as pastas:
CLASS_NAMES = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy',
]

DATA_DIR = "PlantVillage"  # caminho raiz onde estão as pastas acima

# 2) Transformações
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std =[0.229,0.224,0.225]),
])

# 3) Dataset customizado que faz scan de cada classe
class PlantVillageDataset(Dataset):
    def __init__(self, root_dir, class_names, transform=None):
        self.samples = []
        self.transform = transform
        for idx, cls in enumerate(class_names):
            cls_dir = os.path.join(root_dir, cls)
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.png','.jpg','.jpeg')):
                    self.samples.append((os.path.join(cls_dir, fname), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label

def main():
    # Hiperparâmetros
    batch_size = 256
    val_split  = 0.2
    epochs     = 10
    lr         = 1e-3

    # 4) Cria dataset e split
    full_ds = PlantVillageDataset(DATA_DIR, CLASS_NAMES, transform=transform)
    n_val   = int(len(full_ds) * val_split)
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=4)

    # 5) Setup modelo, loss, otimizador
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = ModeloCNN(num_classes=len(CLASS_NAMES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    # 6) Loop de treino
    for ep in range(1, epochs+1):
        model.train()
        worst_loss = 0.0  # inicializa a pior perda para a época
        for batch_idx, (imgs, labels) in enumerate(train_loader, start=1):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # atualiza a pior perda (maior loss)
            if loss.item() > worst_loss:
                worst_loss = loss.item()

            # imprime somente o train loss do batch e pior perda até o momento
            print(f"Ép {ep}/{epochs}  Batch {batch_idx}/{len(train_loader)}  Train Loss: {loss.item():.4f}  -- Pior Perda: {worst_loss:.4f}", end="\r")

        # 7) Validação ao fim da época
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                l = criterion(outputs, labels)
                val_loss += l.item() * imgs.size(0)
        val_loss /= len(val_ds)

        print(f"\n→ Época {ep}  Validation Loss: {val_loss:.4f}")

        # 8) Check-point
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print(f"*** Novo melhor modelo salvo (Val Loss: {val_loss:.4f})\n")

if __name__ == "__main__":
    main()
