import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV
import time
from sklearn.metrics import confusion_matrix, accuracy_score
import os

# Criar uma pasta para salvar as imagens
output_folder = "SVM/results"
os.makedirs(output_folder, exist_ok=True)

dict_rbf = {}
dict_linear = {}

# loading datasets
dists = ['500m', '1km', '1_5km', '2km', '2_5km']
datasets = []
for i in dists:
    distancia_rbf = {}
    distancia_linear = {}
    print(f'distancia ({i}) -------------------------\n')
    dataset = pd.read_csv(f'database{i}.csv')
    datasets.append(dataset)

    X = np.array(dataset.iloc[:, 1::])
    y = np.array(dataset.iloc[:, 0])

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(X, y, stratify=y, test_size=0.2)

    # Randomized Search para encontrar os melhores hiperparâmetros para o kernel RBF
    svc_rbf = SVC(kernel='rbf')
    params_search_rbf = {"C": [0.1, 1, 10, 100, 1000], "gamma": [0.1, 0.01, 0.001, 0.0001]}
    search_rbf = RandomizedSearchCV(svc_rbf, param_distributions=params_search_rbf, n_jobs=-1, cv=3, verbose=5)
    search_rbf.fit(X_train_raw, y_train_raw)
    best_params_rbf = search_rbf.best_params_
    
    # Treinamento do modelo SVM com kernel RBF usando os melhores hiperparâmetros
    start_time = time.time()
    svm_model_rbf = SVC(kernel='rbf', **best_params_rbf)
    svm_model_rbf.fit(X_train_raw, y_train_raw)
    end_time = time.time()
    test_time = end_time - start_time

    # Predição no conjunto de teste para o kernel RBF
    y_pred_test_rbf = svm_model_rbf.predict(X_test_raw)

    # Calcular e armazenar métricas de avaliação para o kernel RBF
    error_rbf = (len(y_test_raw) - sum(y_pred_test_rbf == y_test_raw)) / len(y_test_raw)
    distancia_rbf['Error'] = error_rbf
    distancia_rbf['Accuracy'] = 1 - error_rbf

    precision_rbf = accuracy_score(y_test_raw, y_pred_test_rbf)
    distancia_rbf['Precision'] = precision_rbf

    distancia_rbf['Test Time'] = test_time

    # Matriz de confusão para o kernel RBF
    confusion_rbf = confusion_matrix(y_test_raw, y_pred_test_rbf)
    dist_name = i.replace('_', ',')
    # Salvar a imagem da matriz de confusão para o kernel RBF
    plt.figure()
    plt.imshow(confusion_rbf, cmap='Blues', interpolation='nearest')
    plt.title(f"Matriz de confusão - RBF - {dist_name}")
    plt.colorbar()
    plt.xlabel("Valores Preditos")
    plt.ylabel("Valores verdadeiros")
    plt.xticks(ticks=np.arange(len(confusion_rbf)), labels=np.arange(len(confusion_rbf)))
    plt.yticks(ticks=np.arange(len(confusion_rbf)), labels=np.arange(len(confusion_rbf)))
    plt.savefig(os.path.join(output_folder, f"confusion_matrix_rbf_{i}.png"))
    plt.close()

    dict_rbf[i] = distancia_rbf

    # Randomized Search para encontrar os melhores hiperparâmetros para o kernel linear
    svc_linear = SVC(kernel='linear')
    params_search_linear = {"C": [0.1, 1, 10, 100, 1000]}
    search_linear = RandomizedSearchCV(svc_linear, param_distributions=params_search_linear, n_jobs=-1, cv=3, verbose=5)
    search_linear.fit(X_train_raw, y_train_raw)
    best_params_linear = search_linear.best_params_
    
    # Treinamento do modelo SVM com kernel linear usando os melhores hiperparâmetros
    start_time = time.time()
    svm_model_linear = SVC(kernel='linear', **best_params_linear)
    svm_model_linear.fit(X_train_raw, y_train_raw)
    end_time = time.time()
    test_time = end_time - start_time

    # Predição no conjunto de teste para o kernel linear
    y_pred_test_linear = svm_model_linear.predict(X_test_raw)

    # Calcular e armazenar métricas de avaliação para o kernel linear
    error_linear = (len(y_test_raw) - sum(y_pred_test_linear == y_test_raw)) / len(y_test_raw)
    distancia_linear['Error'] = error_linear
    distancia_linear['Accuracy'] = 1 - error_linear

    precision_linear = accuracy_score(y_test_raw, y_pred_test_linear)
    distancia_linear['Precision'] = precision_linear

    distancia_linear['Test Time'] = test_time

    # Matriz de confusão para o kernel linear
    confusion_linear = confusion_matrix(y_test_raw, y_pred_test_linear)
    # Salvar a imagem da matriz de confusão para o kernel linear
    plt.figure()
    plt.imshow(confusion_linear, cmap='Blues', interpolation='nearest')
    plt.title(f"Matriz de confusão - Linear - {dist_name}")
    plt.colorbar()
    plt.xlabel("Valores Preditos")
    plt.ylabel("Valores verdadeiros")
    plt.xticks(ticks=np.arange(len(confusion_linear)), labels=np.arange(len(confusion_linear)))
    plt.yticks(ticks=np.arange(len(confusion_linear)), labels=np.arange(len(confusion_linear)))
    plt.savefig(os.path.join(output_folder, f"confusion_matrix_linear_{i}.png"))
    plt.close()

    dict_linear[i] = distancia_linear

print("Resultados do kernel RBF:")
print(dict_rbf)

print("Resultados do kernel Linear:")
print(dict_linear)

# Criar DataFrames a partir dos dicionários
df_rbf = pd.DataFrame(dict_rbf).T
df_linear = pd.DataFrame(dict_linear).T

# Escrever os DataFrames em arquivos CSV
df_rbf.to_csv('SVM/output_rbf.csv')
df_linear.to_csv('SVM/output_linear.csv')
