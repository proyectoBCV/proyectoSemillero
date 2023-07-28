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
from tqdm import tqdm
import os
import hashlib
from torch.optim.lr_scheduler import StepLR


start_time = time.time()
# Especificar el dispositivo a utilizar (GPU o CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carga de datos y preprocesamiento
train_data = pd.read_csv('ISIC_2019_Train_data_GroundTruth.csv')
test_data = pd.read_csv('ISIC_2019_Test_data_GroundTruth.csv')
valid_data = pd.read_csv('ISIC_2019_Valid_data_GroundTruth.csv')

torch.manual_seed(42)
random_transforms = [
    transforms.RandomHorizontalFlip(),  # Volteo horizontal
    transforms.RandomVerticalFlip(),    # Volteo vertical
    transforms.RandomRotation(30),      # Rotación aleatoria de hasta 30 grados
    #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # Ajustes aleatorios de color
]

transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionar las imágenes a 224x224 (tamaño requerido por ResNet)
    transforms.ToTensor(),
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
        image_path = f"/media/user_home0/sgoyesp/Proyecto/ISIC_2019_Training_Input/{image_id}.jpg"
        image = Image.open(image_path)
        label = self.data['final_label'].iloc[index]
        if self.transform:
            image = self.transform(image)
        return image, label, image_id


train_dataset = CustomDataset(train_data, transform=transform)
test_dataset = CustomDataset(test_data, transform=transform)
valid_dataset = CustomDataset(valid_data, transform=transform)

# ... Código para crear los data loaders ... AJUSTAR BATCH
batch_train= 443
batch_test = 380
batch_valid = 379


train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_test, shuffle=False, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_valid, shuffle=False, num_workers=4, pin_memory=True)

# Cargar el modelo pre-entrenado ResNet
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Reemplazar la capa completamente conectada para ajustarse al número de clases a 8
model.fc = nn.Linear(num_ftrs, 8)

#Carga de pesos preentrenados en un experimento anterior
#pretrained_state_dict = torch.load("modelo_entrenado2.pth", map_location=device)

# Sacar los pesos de la última capa 
"""
model_state_dict = model.state_dict()
filtered_state_dict = {k: v.to(device) for k, v in pretrained_state_dict.items() if k in model_state_dict}
model_state_dict.update(filtered_state_dict)
model.load_state_dict(model_state_dict)
"""

#Mover el modelo a la GPU
model = model.to(device)

# Asignación de pesos en la función de pérdida 
# Inicializamos todos los pesos en 1
weights = torch.ones(8).to(device)  # Inicializamos todos los pesos en 1
# Definir los pesos específicos para las clases 3, 4, 5 y 7
weights[3] = 5.0
weights[4] = 5.0
weights[5] = 5.0
weights[7] = 5.0
weights.to(device)

# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss(weight=weights)
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
        image_path = f"/media/user_home0/sgoyesp/Proyecto/ISIC_2019_Training_Input/{image_id}.jpg"
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


output_train_directory = "./train_missclassified_images"
output_val_directory = "./val_missclassified_images"
output_test_directory = "./test_missclassified_images"


output_results_file = "results.txt"

with open(output_results_file, 'w') as f:
    for epoch in range(num_epochs):
        #impresión train
        train_predictions, train_labels, t_loss, acc, train_missclassified_images, train_incorrect_labels, train_correct_labels = train(model, train_loader, criterion, optimizer)
        train_precision = precision_score(train_labels, train_predictions, average=None,zero_division=1)
        train_recall = recall_score(train_labels, train_predictions, average=None)
        train_f1_score = f1_score(train_labels, train_predictions, average=None)
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
        valid_recall = recall_score(valid_labels, valid_predictions, average=None)
        valid_f1_score = f1_score(valid_labels, valid_predictions, average=None)
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
        test_recall = recall_score(test_labels, test_predictions, average=None)
        test_f1_score = f1_score(test_labels, test_predictions, average=None)
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

    
torch.save(best_model_state_dict, 'mejor_modelo.pth')

plt.figure()
plt.plot(range(1, num_epochs + 1), train_loss, label='Training Loss', color='blue')
plt.plot(range(1, num_epochs + 1), val_loss, label='Validation Loss', color='green')
plt.plot(range(1, num_epochs + 1), test_loss, label='Test Loss', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss')
plt.legend()
plt.savefig("losses.png")
plt.close()

end_time = time.time()
# Cálculo del tiempo transcurrido
elapsed_time = (end_time - start_time)/60
print(f"Tiempo transcurrido: {elapsed_time} minutos")
