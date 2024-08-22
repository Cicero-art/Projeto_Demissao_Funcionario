import pandas as pd
from pathlib import Path
from analise_bivariada import analise_bivariada  # Importando a função do outro script

# Caminho relativo ao repositório para o arquivo CSV de dados
file_path = Path(__file__).parent.parent / "data" / "HR_comma_sep.csv"
df = pd.read_csv(file_path)

# Caminho para salvar os gráficos
figures_path = Path(__file__).parent.parent / "reports" / "figures"

# Caminho para salvar o sumário CSV
reports_path = Path(__file__).parent.parent / "reports"

# Executar a análise bivariada e gerar o sumário
summary_df = analise_bivariada(df, 'left', figures_path, reports_path)

# Exibindo o sumário na tela
print(summary_df)
