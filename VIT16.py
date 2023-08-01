import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import time
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os
import hashlib
from torch.optim.lr_scheduler import StepLR
from torchvision.models import vision_transformer

start_time = time.time()

device = torch.device("cuda")

train_data = pd.read_csv('ISIC_2019_Train_data_GroundTruth.csv')
test_data = pd.read_csv('ISIC_2019_Test_data_GroundTruth.csv')
valid_data = pd.read_csv('ISIC_2019_Valid_data_GroundTruth.csv')

torch.manual_seed(42)

random_transforms = [
    transforms.RandomHorizontalFlip(),  # Volteo horizontal
    transforms.RandomVerticalFlip(),    # Volteo vertical
    transforms.RandomRotation(30),      # Rotación aleatoria de hasta 30 grados
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Ajustes aleatorios de color
    transforms.RandomPerspective() 
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar las imágenes a 224x224 (tamaño requerido por ResNet)
    transforms.ToTensor(),
    transforms.RandomApply(random_transforms, p=0.5),
    transforms.Normalize((0.5558, 0.5982, 0.6149), (0.2433, 0.1914, 0.1902))  # Normalización de los valores de los píxeles
])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_id = self.data['image'].iloc[index]
        image_path = f"/home/nmercado/data_proyecto/data_proyecto/ISIC_2019_Training_Input/{image_id}.jpg"
        image = Image.open(image_path)
        label = self.data['final_label'].iloc[index]
        if self.transform:
            image = self.transform(image)
        return image, label, image_id


train_dataset = CustomDataset(train_data, transform=transform)
test_dataset = CustomDataset(test_data, transform=transform)
valid_dataset = CustomDataset(valid_data, transform=transform)

# ... Código para crear los data loaders ... AJUSTAR BATCH
batch_train= 50
batch_test = 32
batch_valid = 32


train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_test, shuffle=False, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_valid, shuffle=False, num_workers=4, pin_memory=True)



model = vision_transformer.vit_b_16(weights="ViT_B_16_Weights.IMAGENET1K_V1").to(device)
#print(model)
model.head = nn.Linear(768, 8).to(device)



# Congelar las capas preentrenadas
for name, param in model.named_parameters():
    if "head" not in name:
        param.requires_grad = False

# Asegurarnos de que la nueva capa "head" sea entrenable
for param in model.head.parameters():
    param.requires_grad = True

model = model.to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)


def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    train_predictions = []
    train_labels = []
    missclassified_image_ids = []

    for images, labels, image_ids in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct_predictions += (predicted == labels).sum().item()
        total_samples += labels.size(0)
        train_predictions.extend(predicted.cpu().numpy())
        train_labels.extend(labels.cpu().numpy())
        missclassified_indices = torch.where(predicted != labels)[0]
        #breakpoint()
        incorrect_labels = predicted[missclassified_indices].cpu().numpy()
        correct_labels = labels[missclassified_indices].cpu().numpy()
        missclassified_image_ids.extend([image_ids[idx] for idx in missclassified_indices])
        
    t_loss = train_loss / len(train_loader)
    acc = 100 * correct_predictions / total_samples
    scheduler.step()
    return train_predictions, train_labels, t_loss, acc, missclassified_image_ids, incorrect_labels, correct_labels

#Guardar los pesos del entrenamiento
#torch.save(model.state_dict(), 'modelo_entrenado3.pth')

#Validación y test 
def evaluate(model, data_loader, criterion):
    model.eval()
    loss = 0.0
    correct_predictions = 0
    total_samples = 0
    predictions = []
    labels = []
    missclassified_image_ids = []
    
    with torch.no_grad():
        #breakpoint()
        for images, labels_batch, image_ids in data_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            outputs = model(images)
            batch_loss = criterion(outputs, labels_batch)
            loss += batch_loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels_batch).sum().item()
            total_samples += labels_batch.size(0)
            predictions.extend(predicted.cpu().numpy())
            labels.extend(labels_batch.cpu().numpy())
            #breakpoint()
            missclassified_indices = torch.where(predicted != labels_batch)[0]
            incorrect_labels = predicted[missclassified_indices].cpu().numpy()
            correct_labels = labels_batch[missclassified_indices].cpu().numpy()
            missclassified_image_ids.extend([image_ids[idx] for idx in missclassified_indices.tolist()])
            
            
    avg_loss = loss / len(data_loader)
    accuracy = 100 * correct_predictions / total_samples

    return predictions, labels, avg_loss, accuracy, missclassified_image_ids, incorrect_labels, correct_labels


def guardar_data(ids_missclassified_images, incorrect_labels, correct_labels, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, (image_id, incorrect_label, correct_label) in enumerate(zip(ids_missclassified_images, incorrect_labels, correct_labels)):
        image_path = f"/home/nmercado/data_proyecto/data_proyecto/ISIC_2019_Training_Input/{image_id}.jpg"
        image_pil = Image.open(image_path)
        image_name = f"image_{i}_incorrect_prediction{incorrect_label}_correct_{correct_label}.png"
        image_path = os.path.join(output_dir, image_name)
        image_pil.save(image_path)
    

train_loss=[]
val_loss=[]
test_loss=[]
num_epochs = 10
best_val_loss = float('inf')
best_model_state_dict = None


output_train_directory = "./VIT16train_missclassified_images"
output_val_directory = "./VIT16val_missclassified_images"
output_test_directory = "./VIT16test_missclassified_images"


output_results_file = "results_VIT16_screen_003.txt"

with open(output_results_file, 'w') as f:
    for epoch in range(num_epochs):
        #impresión train
        train_predictions, train_labels, t_loss, acc, train_missclassified_images, train_incorrect_labels, train_correct_labels = train(model, train_loader, criterion, optimizer)
        train_precision = precision_score(train_labels, train_predictions, average=None, zero_division=1)
        train_recall = recall_score(train_labels, train_predictions, average=None, zero_division=1)
        train_f1_score = f1_score(train_labels, train_predictions, average=None, zero_division=1)
        train_loss.append(t_loss)
        if epoch==9:
            guardar_data(train_missclassified_images, train_incorrect_labels, train_correct_labels, output_train_directory)

        print(f'Época: {epoch+1:.4f}', file=f)
        print(f'Training Loss: {t_loss:.4f} | Training Accuracy: {acc:.2f}%', file=f)
        print(f'Training Precision: {train_precision}', file=f)
        print(f'Training Recall: {train_recall}', file=f)
        print(f'Training F1-Score: {train_f1_score}', file=f)
        print('---------------------------', file=f)
        
        #Validación
        valid_predictions, valid_labels, v_loss, v_acc, val_missclassified_images, val_incorrect_labels, val_correct_labels = evaluate(model, valid_loader, criterion) #,valid_images, valid_label_true, valid_label_pred = evaluate(model, valid_loader, criterion)  
        valid_precision = precision_score(valid_labels, valid_predictions, average=None, zero_division=1)
        valid_recall = recall_score(valid_labels, valid_predictions, average=None, zero_division=1)
        valid_f1_score = f1_score(valid_labels, valid_predictions, average=None, zero_division=1)
        valid_accuracy= accuracy_score(valid_labels, valid_predictions, normalize=True)
        val_loss.append(v_loss)
        if epoch==9:
            guardar_data(val_missclassified_images, val_incorrect_labels, val_correct_labels, output_val_directory)
    
        if v_loss < best_val_loss:
            best_val_loss = v_loss
            # Guardar los pesos del modelo en este punto (última capa)
            best_model_state_dict = model.state_dict()

        print('---------- Validación ----------', file=f)
        print(f'Validation Loss: {v_loss:.4f} | Validation Accuracy: {v_acc:.2f}%', file=f)
        print(f'Validation Precision: {valid_precision}', file=f)
        print(f'Validation Recall: {valid_recall}', file=f)
        print(f'Validation F1-Score: {valid_f1_score}', file=f)
        print(f'Validation accuracy: {valid_accuracy}', file=f)
        print('-------------------------------', file=f)

    
        # Evaluación en el conjunto de prueba
        test_predictions, test_labels, t_loss, t_acc, test_missclassified_images, test_incorrect_labels, test_correct_labels = evaluate(model, test_loader, criterion) #, test_images, test_label_true, test_label_pred = evaluate(model, test_loader, criterion)
        test_precision = precision_score(test_labels, test_predictions, average=None, zero_division=1)
        test_recall = recall_score(test_labels, test_predictions, average=None, zero_division=1)
        test_f1_score = f1_score(test_labels, test_predictions, average=None, zero_division=1)
        test_accuracy= accuracy_score(test_labels, test_predictions, normalize=True)
        test_loss.append(t_loss)
        if epoch==9:
            guardar_data(test_missclassified_images, test_incorrect_labels, test_correct_labels, output_test_directory)
    
        print('---------- Prueba ----------', file=f)
        print(f'Test Loss: {t_loss:.4f} | Test Accuracy: {t_acc:.2f}%', file=f)
        print(f'Test Precision: {test_precision}', file=f)
        print(f'Test Recall: {test_recall}', file=f)
        print(f'Test F1-Score: {test_f1_score}', file=f)
        print(f'Test accuracy: {test_accuracy}', file=f)
        print(f"Learning Rate: {scheduler.get_last_lr()[0]}", file=f)
        print('----------------------------', file=f)

    
torch.save(best_model_state_dict, 'mejor_modelo_VIT16_32_screen_003.pth')

plt.figure()
plt.plot(range(1, num_epochs + 1), train_loss, label='Training Loss', color='blue')
plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss', color='green')
plt.plot(range(1, num_epochs + 1), test_loss, label='Test Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.savefig("losses_VIT16_32_screen_003.png")
plt.close()

end_time = time.time()
# Cálculo del tiempo transcurrido
elapsed_time = (end_time - start_time)/60
print(f"Tiempo transcurrido: {elapsed_time} minutos")

