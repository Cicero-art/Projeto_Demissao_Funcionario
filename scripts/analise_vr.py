import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# Definir o caminho do arquivo de forma relativa ao repositório
file_path = Path(__file__).parent.parent / "data" / "HR_comma_sep.csv"

# Carregar somente a coluna 'left' do arquivo CSV
df = pd.read_csv(file_path, usecols=['left'])

# Contar a frequência de cada valor na coluna 'left' e normalizar (convertendo para percentuais)
contagem = df['left'].value_counts(normalize=True) * 100
dist_df = contagem.reset_index()

# Renomear as colunas do DataFrame para nomes mais descritivos
dist_df.columns = ['left', 'Porcentagem']

# Criar um gráfico de barras para visualizar a distribuição da variável 'left'
grafico = sns.barplot(data=dist_df, x='left', y='Porcentagem')

plt.xlabel('Valor')
plt.ylabel('Percentual (%)')
plt.title('Distribuição da Variável target "Left"')

# Adicionar rótulos de porcentagem em cima de cada barra no gráfico
for p in grafico.patches:
    grafico.annotate(
        f'{p.get_height():.1f}%',  # Rótulo formatado para exibir uma casa decimal
        (p.get_x() + p.get_width() / 2., p.get_height()),  # Posição do rótulo
        ha='center',  # Alinhamento horizontal ao centro da barra
        va='center',  # Alinhamento vertical ao centro da barra
        xytext=(0, -10),  # Deslocamento do rótulo em relação ao ponto calculado
        textcoords='offset points',  # Unidades de deslocamento dos rótulos
        fontsize=10  # Tamanho da fonte do rótulo
    )

plt.show()
