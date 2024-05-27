import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score
import os
import time

# Criar uma pasta para salvar as imagens
output_folder = "MLP/results"
os.makedirs(output_folder, exist_ok=True)

results_dict = {}

# Função de treinamento da MLP
def mlp_train(X_train, y_train, hidden_layer_sizes, max_iter=500, tol=1e-4):
    mlp = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, tol=tol)
    mlp.fit(X_train, y_train)
    return mlp

# Função de predição da MLP
def mlp_predict(mlp, X_test):
    return mlp.predict(X_test)

# Carregar os datasets
dists = ['500m', '1km', '1_5km', '2km', '2_5km']
for dist in dists:
    print(f'Dataset: {dist}')
    dataset = pd.read_csv(f'database{dist}.csv')
    
    X = dataset.iloc[:, 1:].values
    y = dataset.iloc[:, 0].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2)

    start_time = time.time()
    mlp = mlp_train(X_train, y_train, hidden_layer_sizes=(100,))  # Defina aqui o número de neurônios em cada camada
    end_time = time.time()
    train_time = end_time - start_time

    start_time = time.time()
    y_pred = mlp_predict(mlp, X_test)
    end_time = time.time()
    test_time = end_time - start_time

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='macro')  # Precisão macro
    confusion = confusion_matrix(y_test, y_pred)
    error = 1 - accuracy  # Calcula o erro como 1 - precisão

    # Salvar a imagem da matriz de confusão
    plt.figure()
    plt.imshow(confusion, cmap='Blues', interpolation='nearest')
    plt.title(f"Matriz de Confusão - {dist.replace('_', ',')}")
    plt.colorbar()
    plt.xlabel("Valores Preditos")
    plt.ylabel("Valores Verdadeiros")
    plt.xticks(ticks=np.arange(len(confusion)), labels=np.arange(len(confusion)))
    plt.yticks(ticks=np.arange(len(confusion)), labels=np.arange(len(confusion)))
    plt.savefig(os.path.join(output_folder, f"confusion_matrix_{dist}.png"))
    plt.close()

    results_dict[dist] = {
        'Error': error,
        'Accuracy': accuracy,
        'Precision': precision,
        'Test Time': train_time+test_time
    }

# Criar um DataFrame a partir do dicionário
results_df = pd.DataFrame(results_dict).T

# Escrever o DataFrame em um arquivo CSV
results_df.to_csv('MLP/output.csv')

print("Resultados salvos em 'MLP/output.csv'.")
