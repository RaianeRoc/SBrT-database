import os
import pandas as pd
import numpy as np


def ler_planilhas_xls(diretorio):
    # Lista para armazenar os caminhos dos arquivos xls
    arquivos_xls = []
    results = []
    # Percorre todos os diretórios e subdiretórios
    for diretorio_raiz, _, arquivos in os.walk(diretorio):
        for arquivo in arquivos:
            # Verifica se o arquivo é do tipo .xls
            if arquivo.endswith('.xls'):
                # Constrói o caminho completo do arquivo
                caminho_arquivo = os.path.join(diretorio_raiz, arquivo)
                # print(caminho_arquivo)
                arquivos_xls.append(caminho_arquivo)

    # Agora temos uma lista com todos os arquivos .xls
    # Vamos tentar ler cada um deles
    for arquivo_xls in arquivos_xls:
        try:
            # Tenta ler o arquivo .xls usando pandas
            # df = pd.read_excel(arquivo_xls, engine='xlrd')
            df = pd.read_csv(arquivo_xls)
            # Aqui você pode processar o DataFrame como desejado
            results.append(df) 
        except Exception as e: pass
    return results

# Diretório onde está localizado o script
diretorio_atual = os.path.dirname(os.path.abspath(__file__))

chuva = ['1km', '1_5km',  '2km', '2_5km', '500m']
intens = [2, 0, 1, 3]
index = 0
indchuva = 0
new_csv = '"Classe de Chuva", "MaxQ", "MinBER", "Attenuation", "Distancia" \n'

# Chamada da função para ler as planilhas .xls no diretório atual e subdiretórios
tabelas = ler_planilhas_xls(diretorio_atual)
for tabela in tabelas:
    maxQ_csv = (tabela.loc[tabela['Layout 1'] == 'Max. Q Factor']).dropna(axis=1)
    minB_csv = (tabela.loc[tabela['Layout 1'] == 'Min. BER']).dropna(axis=1)
    attenuation_csv = (tabela.loc[tabela['Layout 1'] == 'Attenuation']).dropna(axis=1)
    maxQ = (maxQ_csv.to_numpy())[1]
    minB = (minB_csv.to_numpy())[1]
    atten = (attenuation_csv.to_numpy())[0]
    chuva_nome = chuva[indchuva].replace('_','.')

    # if index == 0:
        # new_csv = new_csv + '"classe", "MaxQ", "MinBER", "Attenuation", "Chuva" \n'
    for i in range(1, len(maxQ)):
        new_csv = new_csv + f'{intens[index]}, {float(maxQ[i])}, {float(minB[i])}, {float(atten[i])}, {chuva_nome}\n'


    index += 1
    if index == 4:
        index = 0
        # with open(f'database{chuva[indchuva]}.csv', 'w') as f:
        #     f.write(new_csv)
        # new_csv = ''
        indchuva +=1
print(new_csv)
with open(f'database_todas_as_chuvas.csv', 'w') as f:
    f.write(new_csv)