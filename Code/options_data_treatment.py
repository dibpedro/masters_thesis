import pandas as pd


def options_data_cleaning(df: pd.DataFrame, data_hoje: str) -> pd.DataFrame:
    '''
    Função auxiliar para tratar dados de opções.
    '''

    # Renomeando colunas
    df.columns = df.columns.str.replace(
        '\u00a0', ' ', regex=False).str.strip().str.replace(r'\s+', '_', regex=True)

    # Convertendo os tipos das colunas
    df['Strike'] = pd.to_numeric(df['Strike'], errors='coerce') / 100
    df['Vol_Impl_(%)'] = pd.to_numeric(
        df['Vol_Impl_(%)'], errors='coerce') / 10
    df['Vencimento'] = pd.to_datetime(
        df['Vencimento'], format='%d/%m/%Y', errors='coerce')

    # Removendo NaNs
    df.dropna(subset=['Vencimento', 'Strike', 'Vol_Impl_(%)'], inplace=True)

    # Calculando o tempo até o vencimento
    hoje = pd.Timestamp(data_hoje)
    df['Dias_Vencimento'] = (df['Vencimento'] - hoje).dt.days

    # Selecionando e renomeando as colunas importantes para os gráficos
    colunas = {
        'Tipo': 'tipo',
        'Vencimento': 'vencimento',
        'Strike': 'strike',
        'A/I/OTM': 'moneyness',
        'Vol_Impl_(%)': 'vol_implicita',
        'Dias_Vencimento': 'tau'
    }
    df_novo = df.rename(columns=colunas)

    return df_novo[list(colunas.values())]
