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
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # Ajustes aleatorios de color
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
        image_path = f"/home/nmercado/data_proyecto/data_proyecto/ISIC_2019_Training_Input/{image_id}.jpg"
        image = Image.open(image_path)
        label = self.data['final_label'].iloc[index]
        if self.transform:
            image = self.transform(image)
        return image, label#, image_id


train_dataset = CustomDataset(train_data, transform=transform)
test_dataset = CustomDataset(test_data, transform=transform)
valid_dataset = CustomDataset(valid_data, transform=transform)

# ... Código para crear los data loaders ... AJUSTAR BATCH
batch_train = 443
batch_test = 95
batch_valid = 95

train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_test, shuffle=False, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_valid, shuffle=False, num_workers=4, pin_memory=True)

# Cargar el modelo pre-entrenado ResNet
model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
# Reemplazar la capa completamente conectada para ajustarse al número de clases a 8
model.fc = nn.Linear(num_ftrs, 8)

#Carga de pesos preentrenados en un experimento anterior
pretrained_state_dict = torch.load("modelo_entrenado2.pth", map_location=device)

# Sacar los pesos de la última capa 
model_state_dict = model.state_dict()
filtered_state_dict = {k: v.to(device) for k, v in pretrained_state_dict.items() if k in model_state_dict}
model_state_dict.update(filtered_state_dict)
model.load_state_dict(model_state_dict)

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


#Entrenamiento 
def train(model, train_loader, criterion, optimizer):
    model.train()
    train_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    train_predictions = []
    train_labels = []

    for images, labels in train_loader:
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

    t_loss = train_loss/len(train_loader)
    acc = 100 * correct_predictions/total_samples
    
    return train_predictions, train_labels, t_loss, acc

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
    
    #Lista de imágenes mal clasificadas 
    missclassified_images = []
    #Lista de labels verdaderas de las imagenes mal clasificadas 
    true_labels_missclassified = []
    #Lista de labels mal predichas 
    predicted_labels_missclassified = []
    
    with torch.no_grad():
        for images, labels_batch in data_loader:
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
            missclassified_images.extend(images[missclassified_indices].cpu())
            true_labels_missclassified.extend(images[missclassified_indices].cpu())
            predicted_labels_missclassified.extend(images[missclassified_indices].cpu())
            
    avg_loss = loss / len(data_loader)
    accuracy = 100 * correct_predictions / total_samples
    return predictions, labels, avg_loss, accuracy, missclassified_images, true_labels_missclassified, predicted_labels_missclassified


val_save_dir = './misclassified_valid_images'
os.makedirs(val_save_dir, exist_ok=True)

test_save_dir = './misclassified_test_images'
os.makedirs(test_save_dir, exist_ok=True)

num_epochs = 20
for epoch in range(num_epochs):
    #impresión train
    train_predictions, train_labels, t_loss, acc = train(model, train_loader, criterion, optimizer)
    train_precision = precision_score(train_labels, train_predictions, average=None)
    train_recall = recall_score(train_labels, train_predictions, average=None)
    train_f1_score = f1_score(train_labels, train_predictions, average=None)
    """
    print(f'Época: {epoch+1:.4f}')
    print(f'Training Loss: {t_loss:.4f} | Training Accuracy: {acc:.2f}%')
    print(f'Training Precision: {train_precision}')
    print(f'Training Recall: {train_recall}')
    print(f'Training F1-Score: {train_f1_score}')
    print('---------------------------')"""

    
    #Validación
    valid_predictions, valid_labels, v_loss, v_acc, val_missclassified_images, val_true_labels_missclassified, val_predicted_labels_missclassified = evaluate(model, valid_loader, criterion) #,valid_images, valid_label_true, valid_label_pred = evaluate(model, valid_loader, criterion)  
    valid_precision = precision_score(valid_labels, valid_predictions, average=None)
    valid_recall = recall_score(valid_labels, valid_predictions, average=None)
    valid_f1_score = f1_score(valid_labels, valid_predictions, average=None)
    """
    print('---------- Validación ----------')
    print(f'Validation Loss: {v_loss:.4f} | Validation Accuracy: {v_acc:.2f}%')
    print(f'Validation Precision: {valid_precision}')
    print(f'Validation Recall: {valid_recall}')
    print(f'Validation F1-Score: {valid_f1_score}')
    print('-------------------------------')"""
    
    
    # Evaluación en el conjunto de prueba
    test_predictions, test_labels, t_loss, t_acc, train_missclassified_images, train_true_labels_missclassified, train_predicted_labels_missclassified = evaluate(model, test_loader, criterion) #, test_images, test_label_true, test_label_pred = evaluate(model, test_loader, criterion)
    test_precision = precision_score(test_labels, test_predictions, average=None)
    test_recall = recall_score(test_labels, test_predictions, average=None)
    test_f1_score = f1_score(test_labels, test_predictions, average=None)
    """
    print('---------- Prueba ----------')
    print(f'Test Loss: {t_loss:.4f} | Test Accuracy: {t_acc:.2f}%')
    print(f'Test Precision: {test_precision}')
    print(f'Test Recall: {test_recall}')
    print(f'Test F1-Score: {test_f1_score}')
    print('----------------------------')"""

    for image in val_missclassified_images:
        breakpoint()
        true_label = val_true_labels_missclassified[i].cpu().numpy()#.item()
        predicted_label = val_predicted_labels_missclassified[i].cpu().numpy()#.item()
        image_pil = transforms.ToPILImage()(image)
        #breakpoint()
        # Generar un identificador único utilizando el hash MD5 del contenido de la imagen
        image_hash = hashlib.md5(image.numpy().tobytes()).hexdigest()

        file_path = os.path.join(val_save_dir, f'valid_missclassified_image_{image_hash}_true_{true_label}_predicted_{predicted_label}.jpg')
        image_pil.save(file_path)
        
    
    for i, image in enumerate(train_missclassified_images):
        true_label = train_true_labels_missclassified[i].cpu().numpy()#.item()
        predicted_label = train_predicted_labels_missclassified[i].cpu().numpy()#.item()
        image_pil = transforms.ToPILImage()(image)
        file_path = os.path.join(test_save_dir, f'train_missclassified_image_{i}_true_{true_label}_predicted_{predicted_label}.jpg')
        image_pil.save(file_path)


end_time = time.time()
# Cálculo del tiempo transcurrido
elapsed_time = (end_time - start_time)/60
print(f"Tiempo transcurrido: {elapsed_time} minutos")
