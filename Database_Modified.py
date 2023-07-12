import numpy as np
import pandas as pd

dic = {'MEL': 0, 'NV': 1, 'BCC': 2, 'AK': 3, 'BKL': 4, 'DF': 5, 'VASC': 6, 'SCC': 7, 'UNK': 8}

data = pd.read_csv('ISIC_2019_Training_GroundTruth.csv')
data['final_label'] = ''

for columna in data.columns[1:-1]:
    n_column = columna
    for indice, valor in data[columna].items():
        if valor == 1:
            n_fila = data.loc[indice, 'image']
            num_lab = dic[n_column]
            data.loc[indice, 'final_label'] = num_lab

new_data = data[['image', 'final_label']]
new_data.to_csv('ISIC_2019_Final_GroundTruth.csv', index=False)
