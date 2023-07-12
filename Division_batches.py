import numpy as np
import pandas as pd

new_data = pd.read_csv('ISIC_2019_Final_GroundTruth_Binary.csv')

numeros_unicos = new_data['final_label'].unique()
porcentaje_1 = 0.7
porcentaje_2 = 0.15

train_data = pd.DataFrame()
valid_data = pd.DataFrame()
test_data = pd.DataFrame()
imagenes_unicas = set()


for numero in numeros_unicos:
    datos_filtrados = new_data[new_data['final_label'] == numero]
    total_filas = len(datos_filtrados)
    filas_lote_1 = int(total_filas * porcentaje_1)
    filas_lote_2 = int(total_filas * porcentaje_2)
    
    datos_filtrados = datos_filtrados[~datos_filtrados['image'].isin(imagenes_unicas)]

    train_subset = datos_filtrados[:filas_lote_1]
    valid_subset = datos_filtrados[filas_lote_1:filas_lote_1+filas_lote_2]
    test_subset = datos_filtrados[filas_lote_1+filas_lote_2:]
    
    # Actualizar el conjunto de imágenes únicas con las nuevas imágenes
    imagenes_unicas.update(datos_filtrados['image'])
    
    train_data = pd.concat([train_data, train_subset], ignore_index=True)
    valid_data = pd.concat([valid_data, valid_subset], ignore_index=True)
    test_data = pd.concat([test_data, test_subset], ignore_index=True)

train_data.to_csv('ISIC_2019_Train_data_GroundTruth_Binary.csv', index=False)
valid_data.to_csv('ISIC_2019_Valid_data_GroundTruth_Binary.csv', index=False)
test_data.to_csv('ISIC_2019_Test_data_GroundTruth_Binary.csv', index=False)
