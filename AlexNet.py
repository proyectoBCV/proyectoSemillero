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
from torchvision.models import alexnet
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
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)  # Ajustes aleatorios de color
]

transform = transforms.Compose([
    transforms.Resize((227, 227)), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # Normalize using ImageNet statistics
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
        return image, label#, image_id


train_dataset = CustomDataset(train_data, transform=transform)
test_dataset = CustomDataset(test_data, transform=transform)
valid_dataset = CustomDataset(valid_data, transform=transform)

# ... Código para crear los data loaders ... AJUSTAR BATCH
batch_train = 128  
batch_test = 64
batch_valid = 64


train_loader = DataLoader(train_dataset, batch_size=batch_train, shuffle=True, num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=batch_test, shuffle=False, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_valid, shuffle=False, num_workers=4, pin_memory=True)

# Cargar el modelo pre-entrenado AlexNet
model = alexnet(pretrained=True)
num_ftrs = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_ftrs, 8)  

model_state_dict = model.state_dict()


model = model.to(device)
weights = torch.ones(8).to(device) 

# Definir los pesos específicos para las clases 3, 5 y 7
weights[3] = 5.0
weights[4] = 5.0
weights[5] = 5.0
weights[7] = 5.0
weights.to(device)
# Definir la función de pérdida y el optimizador
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = StepLR(optimizer, step_size=3, gamma=0.1)



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
    scheduler.step()
    #print(f"Epoch {epoch+1}/{num_epochs}, Learning Rate: {scheduler.get_last_lr()[0]}")
    return train_predictions, train_labels, t_loss, acc

torch.save(model.state_dict(), 'modelo_entrenado4.pth')

#Validación y test 
def evaluate(model, data_loader, criterion):
    model.eval()
    loss = 0.0
    correct_predictions = 0
    total_samples = 0
    predictions = []
    labels = []
    sample_images = []
    sample_labels_true = []
    sample_labels_pred = []

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

    avg_loss = loss / len(data_loader)
    accuracy = 100 * correct_predictions / total_samples
    return predictions, labels, avg_loss, accuracy #, sample_images, sample_labels_pred, sample_labels_true


num_epochs = 10
for epoch in range(num_epochs):
    #impresión train
    train_predictions, train_labels, t_loss, acc = train(model, train_loader, criterion, optimizer)
    train_precision = precision_score(train_labels, train_predictions, average=None)
    train_recall = recall_score(train_labels, train_predictions, average=None)
    train_f1_score = f1_score(train_labels, train_predictions, average=None)

    print(f'Época: {epoch+1:.4f}')
    print(f'Training Loss: {t_loss:.4f} | Training Accuracy: {acc:.2f}%')
    print(f'Training Precision: {train_precision}')
    print(f'Training Recall: {train_recall}')
    print(f'Training F1-Score: {train_f1_score}')
    print('---------------------------')

       
    
    #Validación
    valid_predictions, valid_labels, v_loss, v_acc = evaluate(model, valid_loader, criterion) #,valid_images, valid_label_true, valid_label_pred = evaluate(model, valid_loader, criterion)  
    valid_precision = precision_score(valid_labels, valid_predictions, average=None)
    valid_recall = recall_score(valid_labels, valid_predictions, average=None)
    valid_f1_score = f1_score(valid_labels, valid_predictions, average=None)

    print('---------- Validación ----------')
    print(f'Validation Loss: {v_loss:.4f} | Validation Accuracy: {v_acc:.2f}%')
    print(f'Validation Precision: {valid_precision}')
    print(f'Validation Recall: {valid_recall}')
    print(f'Validation F1-Score: {valid_f1_score}')
    print('-------------------------------')
    
    # Evaluación en el conjunto de prueba
    test_predictions, test_labels, t_loss, t_acc = evaluate(model, test_loader, criterion) #, test_images, test_label_true, test_label_pred = evaluate(model, test_loader, criterion)
    test_precision = precision_score(test_labels, test_predictions, average=None)
    test_recall = recall_score(test_labels, test_predictions, average=None)
    test_f1_score = f1_score(test_labels, test_predictions, average=None)

    print('---------- Prueba ----------')
    print(f'Test Loss: {t_loss:.4f} | Test Accuracy: {t_acc:.2f}%')
    print(f'Test Precision: {test_precision}')
    print(f'Test Recall: {test_recall}')
    print(f'Test F1-Score: {test_f1_score}')
    print('----------------------------')


end_time = time.time()
# Cálculo del tiempo transcurrido
elapsed_time = (end_time - start_time)/60
print(f"Tiempo transcurrido: {elapsed_time} minutos")



end_time = time.time()
# Cálculo del tiempo transcurrido
elapsed_time = (end_time - start_time)/60
print(f"Tiempo transcurrido: {elapsed_time} minutos")
