import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler
import os


def GBM_simulation(s0: float, mu: float, sigma: float, T: float, N: int, num_sim: int = 1) -> np.ndarray:
    '''Simula num_sim trajetórias de um MB geométrico (GBM).

    Recebe como argumentos:
    s0 -> Preço inicial do ativo
    mu -> Coeficiente de drift
    sigma -> Volatilidade
    T -> Horizonte de tempo da simulação
    N -> Números de passos de tempo (discretização)
    num_sim -> O número de trajetórias a serem simuladas

    Retorna uma matriz (N+1, num_sim) com as trajetórias simuladas.
    '''

    dt = T/N    # tamanho do passo de tempo

    # Matriz para armazenar as trajetórias
    S = np.zeros((N+1, num_sim))
    # definindo a primeira como S_0 para todas as trajetórias ("linha 0, todas as colunas")
    S[0, :] = s0

    # Cria uma matriz aleatória N x num_sim para todas as trajetórias de uma vez
    W = np.random.standard_normal((N, num_sim))

    # Simulando os passos de tempo
    for t in range(1, N+1):
        S[t, :] = S[t-1, :] * \
            np.exp((mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt) * W[t-1, :])

    return S


######################################## Execução de teste ################################################

if __name__ == "__main__":

    # Parâmetros da simulação
    s0 = 100
    mu = 0.05
    sigma = 0.20
    T = 1
    N = 252
    num_sim = 5

    # Simulação
    trajetorias = GBM_simulation(s0, mu, sigma, T, N, num_sim)

    # Plotando o gráfico
    tempo = np.linspace(0, T, N+1)
    fig, ax = plt.subplots(figsize=(11, 7))

    # Configurando a paleta de cores
    cmap = plt.get_cmap('PuRd')
    cores = cmap(np.linspace(0.3, 1, num_sim))

    ciclo_cores = cycler(color=cores)
    ax.set_prop_cycle(ciclo_cores)

    plt.plot(tempo, trajetorias, alpha=0.7, linewidth=1.5)

    plt.title(f'{num_sim} Trajetórias Simuladas do MBG', fontsize=13)
    plt.xlabel('Tempo', fontsize=11)
    plt.ylabel('Preço do Ativo S(t)', fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.5)

    output_dir = 'Figuras'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    caminho = os.path.join(output_dir, 'GBM.pdf')
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.show()
