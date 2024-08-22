import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def analise_bivariada(dataset, target, figures_path, reports_path):
    """
    Gera análises bivariadas e um sumário para a varável target a partir de um dataset.

    Parameters
    ----------
    dataset: pd.DataFrame
        Uma tabela contendo colunas com variáveis e uma variável alvo target.
    target: str
        Nome da coluna que representa a variável resposta.
    figures_path: str
        Caminho onde serão salvos os gráficos gerados.
    reports_path: str
        Caminho onde será salvo o CSV com o sumário.

    Returns
    -------
    pd.DataFrame
        DataFrame contendo o sumário das distribuições percentuais.
    """

    # Verificação de parâmetros
    if not isinstance(dataset, pd.DataFrame):
        raise TypeError('O dataset deve ser do tipo pd.DataFrame')
    if not isinstance(target, str):
        raise TypeError('O target deve ser do tipo str')
    if not isinstance(figures_path, str):
        raise TypeError('O figures_path deve ser do tipo str')
    if not isinstance(reports_path, str):
        raise TypeError('O reports_path deve ser do tipo str')

    dataset.drop_duplicates(inplace=True)
    discrete_columns = [col for col in dataset.columns if
                        dataset[col].nunique() <= 10 and dataset[col].dtype in ['int64', 'float64']]

    # Criar a pasta para figuras, se não existir
    if not os.path.exists(figures_path):
        os.makedirs(figures_path)

    summary_data = []

    for coluna in dataset.columns:
        if coluna != target:
            plt.figure(figsize=(10, 6))

            if coluna in discrete_columns and target in discrete_columns:
                # Adicionando ao sumário as porcentagens_
                prop_df = dataset.groupby(coluna)[target].value_counts(normalize=True).unstack()
                prop_df = prop_df * 100  # Converte para porcentagem

                for category in prop_df.index:
                    for val in prop_df.columns:
                        summary_data.append({
                            'Variable': coluna,
                            'Category': category,
                            f'{target}={val} (%)': prop_df.loc[category, val]
                        })

                sns.countplot(data=dataset, x=coluna, hue=target)
                plt.xlabel(coluna)
                plt.ylabel('Contagem')
                plt.title(f'Distribuição de {target} por {coluna}')
                filename = f"{coluna}_countplot.png"

            elif dataset[coluna].dtype == 'object' or coluna in discrete_columns:
                # Adicionando ao sumário as porcentagens
                prop_df = dataset.groupby(coluna)[target].value_counts(normalize=True).unstack()
                prop_df = prop_df * 100  # Converte para porcentagem

                for category in prop_df.index:
                    for val in prop_df.columns:
                        summary_data.append({
                            'Variable': coluna,
                            'Category': category,
                            f'{target}={val} (%)': prop_df.loc[category, val]
                        })

                sns.countplot(data=dataset, x=coluna, hue=target)
                plt.xlabel(coluna)
                plt.ylabel('Contagem')
                plt.title(f'Distribuição de {target} por {coluna}')
                filename = f"{coluna}_countplot.png"

            elif dataset[coluna].dtype in ['int64', 'float64']:
                fig, ax = plt.subplots(1, 2, figsize=(10, 6))
                sns.histplot(data=dataset, x=coluna, hue=target, multiple='stack', ax=ax[0])
                ax[0].set_title(f'Histograma de {coluna} por {target}')
                sns.boxplot(data=dataset, x=target, y=coluna, ax=ax[1])
                ax[1].set_title(f'Boxplot de {target} por {coluna}')
                filename = f"{coluna}_scatter_boxplot.png"

            plt.savefig(os.path.join(figures_path, filename))
            plt.close()

    # Convertendo os dados do sumário em um DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Salvando o sumário em um arquivo CSV na pasta reports
    summary_df.to_csv(os.path.join(reports_path, 'summary_distribution.csv'), index=False)

    return summary_df
