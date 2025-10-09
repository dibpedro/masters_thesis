import numpy as np
import matplotlib.pyplot as plt
import os


def BM_increments(T: float, passos: int, num_traj: int) -> np.ndarray:
    '''Gera incrementos independentes de MBs padrão.
    Recebe como argumentos:
    T -> Horizonte de tempo final
    passos -> Número de passos de tempo na discretização
    num_traj -> Número de trajetórias a serem criadas

    Retorna uma matriz (passos, num_traj) com os incrementos dW
    '''

    dt = T / passos
    dW = np.random.normal(loc=0.0, scale=np.sqrt(dt), size=(passos, num_traj))
    return dW


def correlated_BMs(T: float, passos: int, rho: float) -> tuple:
    '''
    Simula 2 trajetórias de MBs correlacionados a partir de MBs independentes através da decomposição de Cholesky.
    Recebe como argumentos:
    T -> Horizonte de tempo final
    passos -> Número de passos de tempo na discretização
    rho -> Coeficiente de correlação

    Retorna uma tupla contendo (tempo, W1, W2, dW1, dW2)
    '''

    # Gerando 2 incrementos independentes através da função BM_increments
    dB = BM_increments(T, passos, 2)

    # Criando a matriz de correlação e aplicando Cholesky
    C = np.array([[1.0, rho],
                  [rho, 1.0]])
    L = np.linalg.cholesky(C)

    # Transformar os incrementos indep. em correl.
    dW_correl = dB @ L.T
    dW1 = dW_correl[:, 0]
    dW2 = dW_correl[:, 1]

    # Calculando as trajetórias e acrescentando W(0) = 0
    W1, W2 = np.cumsum(dW1), np.cumsum(dW2)
    W1, W2 = np.insert(W1, 0, 0), np.insert(W2, 0, 0)

    tempo = np.linspace(0, T, passos + 1)

    return tempo, W1, W2, dW1, dW2


# cores rho > 0: mediumturquoise e mediumvioletred
# cores rho < 0: mediumpurple e navy

def correlated_BM_plot(tempo: np.ndarray, W1: np.ndarray, W2: np.ndarray, rho: float):
    """Função auxiliar para plotar os dois MBs."""
    plt.figure(figsize=(11, 7))
    plt.plot(tempo, W1, label='$W_1$', color='mediumpurple')
    plt.plot(tempo, W2, label='$W_2$', color='navy')
    plt.title(f'ρ={rho}', fontsize=24, fontweight='bold')
    plt.xlabel('t', fontsize=20)
    plt.ylabel('W(t)', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=14)
    plt.grid(True)

    output_dir = 'Figuras'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    nome = f'MB_correlacionados_rho_{rho}.pdf'
    caminho = os.path.join(output_dir, nome)
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.show()


#################################### Execução de teste #############################################

if __name__ == "__main__":

    # Parâmetros da simulação
    T = 1
    passos = 252
    rho = -0.7

    # Simulando os dados
    tempo, W1, W2, dW1, dW2 = correlated_BMs(T, passos, rho)

    # Plotando o resultado
    correlated_BM_plot(tempo, W1, W2, rho)
