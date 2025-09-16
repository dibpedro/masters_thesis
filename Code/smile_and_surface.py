import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_smile(ativo: str, tipo: str, vencimento: str, df: pd.DataFrame):
    '''
    Função para gerar a curva smile de um ativo.
    Inputs:
    ativo -> Nome do ativo, por exemplo, VALE3
    tipo -> O tipo da opção: call ou put
    vencimento -> Vencimento da opção, por exemplo: '2025-10-11'
    df -> Dataframe com os dados necessários.
    Retorna o gráfico em questão.
    '''

    # Definindo o vencimento
    venc = df[df['vencimento'] == vencimento].copy()
    if venc.empty:
        print('Não foram encontrados dados para esse vencimento')
        return

    # Escolhendo o tipo de opção
    if tipo == 'call':
        dataf = venc[venc['tipo'] == 'CALL']
    elif tipo == 'put':
        dataf = venc[venc['tipo'] == 'PUT']
    else:
        print('Tipo de opção não disponível')
        return

    # Gerando o gráfico
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(x=dataf['strike'], y=dataf['vol_implicita'],
               color='navy', label=tipo.upper(), s=80, alpha=0.7)

    # Customizando
    ax.set_title(f'{ativo} - Vencimento: {vencimento}', fontsize=14, pad=20)
    ax.set_xlabel('Strike', fontsize=12)
    ax.set_ylabel('Volatilidade Implícita (%)', fontsize=12)
    ax.grid(True)

    ax.legend()
    plt.savefig(f'smile_{ativo} - {vencimento}.pdf', bbox_inches='tight')
    plt.show()


def vol_surface(ativo: str, data_hoje: str, df: pd.DataFrame):
    '''Função para gerar a superfície de volatilidade implícita de um ativo.
    Inputs:
    ativo -> Nome do ativo, por exemplo, VALE3
    data_hoje -> A data de início da análise, exemplo: '2025-08-29'
    df -> Dataframe com os dados necessários.
    Retorna o gráfico em questão.
    '''

    # Removendo possíveis NaN
    df.dropna(subset=['strike', 'tau', 'vol_implicita'], inplace=True)

    # Criando o gráfico e o eixo
    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(projection='3d')

    scatter = ax.scatter(df['strike'], df['tau'], df['vol_implicita'],
                         c=df['vol_implicita'], cmap='coolwarm', s=40, alpha=0.8)

    # Customizando
    ax.set_title(f'{ativo}', fontsize=15, pad=7)
    ax.set_xlabel('Strike', fontsize=12, labelpad=10)
    ax.set_ylabel('$\tau$ - Dias até o Vencimento', fontsize=12, labelpad=10)
    ax.set_zlabel('Volatilidade Implícita (%)', fontsize=12, labelpad=10)
    fig.colorbar(scatter, shrink=0.5, aspect=10, label='Vol. Implícita (%)')

    ax.view_init(elev=45, azim=-50)
    plt.savefig(f'SVI_{ativo} - {data_hoje}.pdf', bbox_inches='tight')
    plt.show()
