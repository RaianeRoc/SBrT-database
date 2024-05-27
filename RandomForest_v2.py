import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# Criar uma pasta para salvar as imagens
output_folder = "RandomForest/results"
os.makedirs(output_folder, exist_ok=True)

# Dicionário para armazenar os resultados
dict_rf = {}

# Carregar os datasets
dists = ['500m', '1km', '1_5km', '2km', '2_5km']
for i in dists:
    print(f'distancia ({i}) -------------------------\n')
    dataset = pd.read_csv(f'database{i}.csv')
    
    X = dataset.iloc[:, 1:]
    y = dataset.iloc[:, 0]

    # Dividir o conjunto de dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    # Inicializar o classificador Random Forest
    rf = RandomForestClassifier(random_state=42)

    # Treinar o modelo
    start_time = time.time()
    rf.fit(X_train, y_train)
    end_time = time.time()
    test_time = end_time - start_time

    # Fazer previsões no conjunto de teste
    y_pred_test = rf.predict(X_test)

    # Calcular métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred_test)
    error = 1 - accuracy
    precision = accuracy_score(y_test, y_pred_test)

    # Calcular a matriz de confusão
    confusion = confusion_matrix(y_test, y_pred_test)

    # Nome da distância para fins de salvamento
    dist_name = i.replace('_', ',')

    # Salvar a imagem da matriz de confusão
    plt.figure()
    plt.imshow(confusion, cmap='Blues', interpolation='nearest')
    plt.title(f"Matrix de confusão - {dist_name}")
    plt.colorbar()
    plt.xlabel("Valores Preditos")
    plt.ylabel("Valores verdadeiros")
    plt.xticks(ticks=np.arange(len(confusion)), labels=np.arange(len(confusion)))
    plt.yticks(ticks=np.arange(len(confusion)), labels=np.arange(len(confusion)))
    plt.savefig(os.path.join(output_folder, f"confusion_matrix_{i}.png"))
    plt.close()

    # Armazenar os resultados no dicionário
    dict_rf[i] = {
        'Error': error,
        'Accuracy': accuracy,
        'Precision': precision,
        'Test Time': test_time
    }

# Criar um DataFrame a partir do dicionário
df_rf = pd.DataFrame(dict_rf).T

# Escrever o DataFrame em um arquivo CSV
df_rf.to_csv('RandomForest/output.csv')

print("Resultados salvos com sucesso!")
