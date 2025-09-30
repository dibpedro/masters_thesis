import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
import os


def heston_paths(params: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    '''Simula trajetórias do modelo de Heston a partir da discretização de Milstein.
    Recebe como parâmetros:
    params - Um dicionário contendo todos os parâmetros necessários para a simulação. São eles:
            S0 -> preço inicial do ativo
            v0 -> variância inicial
            r -> taxa de juros livre de risco
            kappa -> taxa de reversão à média
            v_bar -> média a longo prazo da variância
            gamma -> volatilidade da volatilidade
            rho -> coeficiente de correlação
            T -> horizonte de tempo
            num_steps -> número de passos na simulação
            num_sim -> número de trajetórias a serem simuladas

    Retorna uma tupla contendo duas matrizes numpy: uma para a trajetória dos preços e outra para a trajetória das variâncias.
    '''

    # Desempacotando os parâmetros para facilitar
    S0 = params['S0']
    v0 = params['v0']
    r = params['r']
    kappa = params['kappa']
    v_bar = params['v_bar']
    gamma = params['gamma']
    rho = params['rho']
    T = params['T']
    num_steps = params['num_steps']
    num_sims = params['num_sims']

    # Preparando a simulação
    dt = T / num_steps                              # o tamanho de cada passo no tempo
    # cria uma matriz vazia para as trajetórias
    S = np.zeros((num_steps + 1, num_sims))
    # cada linha representa um passo no tempo e cada coluna uma simulação
    v = np.zeros((num_steps + 1, num_sims))
    S[0, :] = S0                                    # valores iniciais
    v[0, :] = v0

    # Simulação
    for t in range(1, num_steps + 1):
        # Gerando os incrementos brownianos
        Z1 = np.random.normal(size=num_sims)
        Z2 = np.random.normal(size=num_sims)

        # Correlacionando-os
        dW1 = np.sqrt(dt) * Z1
        dW2 = np.sqrt(dt) * (rho * Z1 + np.sqrt(1 - rho**2) * Z2)

        # Hipótese de absorção
        v_prev_pos = np.maximum(v[t-1, :], 0)

        # Implementando o esquema de Milstein
        v_t = (v[t-1, :] +
               kappa*(v_bar - v_prev_pos)*dt +
               gamma*np.sqrt(v_prev_pos)*dW2 +
               0.25*gamma**2*(dW2**2 - dt))
        v[t, :] = np.maximum(v_t, 0)

        # Log-preço do ativo
        ln_St = np.log(S[t-1, :]) + (r - 0.5*v_prev_pos) * \
            dt + np.sqrt(v_prev_pos)*dW1
        S[t, :] = np.exp(ln_St)

    return S, v


def plot_heston(paths: np.ndarray, T: float, num_steps: int, title: str, ylabel: str, color: str, num_traj: int) -> None:
    '''
    Função auxiliar para plotar as trajetórias obtidas através da função heston_paths().
    Recebe como parâmetros:
    paths -> matriz de trajetórias (preço ou variância)
    T -> horizonte de tempo
    num_steps -> número de passos na simulação
    title -> título do gráfico
    ylabel -> nome do eixo y
    color -> nome do colormap a ser utilizado
    num_traj -> número de trajetórias a serem exibidas no gráfico

    Retorna as imagens dos gráficos
    '''

    tempo = np.linspace(0, T, num_steps + 1)                   # eixo temporal
    # escolhe o colormap
    cmap = plt.get_cmap(color)
    colors = [cmap(i) for i in np.linspace(0.2, 1, num_traj)]  # lista de cores

    # Gerando a figura
    plt.figure(figsize=(10, 6))

    # Plotando cada trajetória com uma cor da lista
    for i in range(num_traj):
        plt.plot(tempo, paths[:, i], color=colors[i])

    plt.title(title, fontsize=17)
    plt.ylabel(ylabel, fontsize=11)
    plt.xlabel('t', fontsize=11)
    plt.grid(True)
    plt.legend()

    output_dir = 'Figuras'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    caminho = os.path.join(output_dir, f'{ylabel}_heston.pdf')
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.show()


############################################################ Execução de teste ###################################################################
if __name__ == "__main__":

    # Parâmetros do modelo e da simulação
    params = {
        'S0': 100.0,
        'v0': 0.04,
        'r': 0.05,
        'kappa': 2.0,
        'v_bar': 0.04,
        'gamma': 0.3,
        'rho': -0.7,
        'T': 1.0,
        'num_steps': 252,
        'num_sims': 1000
    }

    S_paths, v_paths = heston_paths(params)
    plot_heston(v_paths, params['T'], params['num_steps'],
                'Processo de variância $v_t$ - Heston', '$v_t$', 'GnBu', 7)
