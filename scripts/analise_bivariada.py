import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("C:/Users/cicero.neto/Projeto_Demissao_Funcionario/data/HR_comma_sep.csv")


def analise_bivariada(dataset, target, path):

    dataset.drop_duplicates(inplace=True)
    # Identificar colunas que são variáveis discretas, incluindo variáveis binárias
    discrete_columns = [col for col in dataset.columns if
                        dataset[col].nunique() <= 10 and dataset[col].dtype in ['int64', 'float64']]
    # Verificar se existe a pasta destino, caso não exista, cria uma
    if not os.path.exists(path):
        os.makedirs(path)

    for coluna in dataset.columns:

        # Verifcação ignorando a coluna alvo
        if coluna != target:
            plt.figure(figsize=(10, 6))

            # Caso de 2 variáveis discretas
            if coluna in discrete_columns and target in discrete_columns:

                # Gráfico de barras agrupadas para duas variáveis categóricas
                sns.countplot(data=dataset, x=coluna, hue=target)
                plt.xlabel(coluna)
                plt.ylabel('Contagem')
                plt.title(f'Distribuição de {target} por {coluna}')
                filename = f"{coluna}_countplot.png"

            # Caso de 1 variavél discreta e 1 contínua
            elif dataset[coluna].dtype == 'object' or coluna in discrete_columns:
                # Gráfico de barras padrão para variáveis categóricas
                sns.countplot(data=dataset, x=coluna, hue=target)
                plt.xlabel(coluna)
                plt.ylabel('Contagem')
                plt.title(f'Distribuição de {target} por {coluna}')
                filename = f"{coluna}_countplot.png"

            # Caso de 2 variáveis contínuas
            elif dataset[coluna].dtype in ['int64', 'float64']:
                # Histograma e Boxplot para 2 variáveis contínuas
                fig, ax = plt.subplots(1,2, figsize=(10, 6))
                sns.histplot(data=dataset, x=coluna, hue=target, multiple='stack', ax=ax[0])
                ax[0].set_title(f'Histograma de {coluna} por {target}')
                sns.boxplot(data=dataset, x=target, y=coluna, ax=ax[1])
                ax[1].set_title(f'Boxplot de {target} por {coluna}')
                filename = f"{coluna}_scatter_boxplot.png"
            # salvar figura no "path" com o nome "filename"
            plt.savefig(os.path.join(path, filename))

analise_bivariada(df, 'left', "C:/Users/cicero.neto/Projeto_Demissao_Funcionario/data")
