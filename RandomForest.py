import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
import time

def run(data, distancia):
    # Dividir em características e rótulos
    X = data.iloc[:, 1:]  # Características
    y = data.iloc[:, 0]   # Classes

    # Dividir em conjunto de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Inicializar o classificador Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Treinar o modelo
    start_time = time.time()
    rf.fit(X_train, y_train)
    end_time = time.time()

    # Fazer previsões
    y_pred = rf.predict(X_test)

    # Calcular métricas de avaliação
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=1) # Definir zero_division=1

    # Imprimir as métricas
    print("Accuracy:", accuracy)
    print("F1 Score (Weighted):", f1)
    print("Recall (Weighted):", recall)
    print("Precision (Weighted):", precision)

    # Tempo de teste
    test_time = end_time - start_time
    print("Test Time:", test_time)

    # Reduzir a dimensionalidade para visualização
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)

    # Identificar as previsões corretas e incorretas
    correct_predictions = (y_test == y_pred)
    incorrect_predictions = ~correct_predictions

    # Configurar subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Plotar distribuição das classes
    for label in y_test.unique():
        axes[0].scatter(X_pca[y_test == label, 0], X_pca[y_test == label, 1], label=f'Classe {label}')
    axes[0].set_title(f'Distribuição das Classes para {distancia}')
    axes[0].set_xlabel('Min BER')
    axes[0].set_ylabel('Max. Q Factor')
    axes[0].legend()

    # Plotar erros de classificação
    axes[1].scatter(X_pca[correct_predictions, 0], X_pca[correct_predictions, 1], label='Correto', color='blue')
    axes[1].scatter(X_pca[incorrect_predictions, 0], X_pca[incorrect_predictions, 1], label='Incorreto', color='red')
    axes[1].set_title(f'Erros de Classificação para {distancia}')
    axes[1].set_xlabel('Min BER')
    axes[1].set_ylabel('Max. Q Factor')
    axes[1].legend()

    # Adicionar métricas no título dos subplots
    axes[0].set_title(f'Distribuição das Classes\nAccuracy: {accuracy:.2f}, F1 Score: {f1:.2f}, Recall: {recall:.2f}, Precision: {precision:.2f}')
    axes[1].set_title(f'Erros de Classificação\nTest Time: {test_time:.2f}')

    # Ajustar o layout
    plt.tight_layout()

    # Salvar a imagem
    # plt.savefig(f'analise_RF_{distancia}.png')

    # Exibir o gráfico
    plt.show()

# Exemplo de uso:
run(pd.read_csv('database1_5km.csv'), '1500m')
run(pd.read_csv('database2_5km.csv'), '2500m')
run(pd.read_csv('database2km.csv'), '2000m')
run(pd.read_csv('database1km.csv'), '1000m')
run(pd.read_csv('database500m.csv'), '500m')