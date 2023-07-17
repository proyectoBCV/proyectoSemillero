
import pandas as pd

data = pd.read_csv('ISIC_2019_Train_data_GroundTruth.csv')
# Contar el número de etiquetas en final_label
etiqueta_contador = data['final_label'].value_counts()

data2 = pd.read_csv('ISIC_2019_Test_data_GroundTruth.csv')
et = data2['final_label'].value_counts()

data3 = pd.read_csv('ISIC_2019_Valid_data_GroundTruth.csv')
et2 = data3['final_label'].value_counts()


# Mostrar los resultados
print("Contador train : ", etiqueta_contador)
print("Contador test : ", et)
print("Contador valid :", et2)
"""
import pandas as pd
from sklearn.model_selection import train_test_split

# Cargar los datos
data = pd.read_csv('ISIC_2019_Final_GroundTruth.csv')

# Dividir los datos en características (X) y etiquetas (y)
X = data['image']
y = data['final_label']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear DataFrames para los conjuntos de entrenamiento y prueba
train_data = pd.DataFrame({'image_id': X_train, 'final_label': y_train})
test_data = pd.DataFrame({'image_id': X_test, 'final_label': y_test})

# Guardar los DataFrames en archivos CSV
train_data.to_csv('train.csv', index=False)
test_data.to_csv('test.csv', index=False)"""
