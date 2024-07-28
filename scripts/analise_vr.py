import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Ler o arquivo do diretório
df = pd.read_csv("C:/Users/cicero.neto/Projeto_Demissao_Funcionario/data/HR_comma_sep.csv")
print(type(df))
# Contar, normalizar e adicioncar um dataframe para visualização da distribuição
contagem = df['left'].value_counts(normalize=True) * 100
dist_df = contagem.reset_index()
dist_df.columns = ['left', 'Porcentagem']

# Visualização da distribuição
grafico = sns.barplot(data=dist_df, x='left', y='Porcentagem')
plt.xlabel('valor')
plt.ylabel('Percentual (%)')
plt.title('Distribuição da Variável target "Left"')

for p in grafico.patches:
    # Adicionar o rótulo de dados
    grafico.annotate(
        f'{p.get_height():.1f}%',
        (p.get_x() + p.get_width() / 2.,
         p.get_height()),
        ha='center',
        va='center',
        xytext=(0, -10),
        textcoords='offset points',
        fontsize=10
    )

plt.show()
