import numpy as np
import matplotlib.pyplot as plt
import os


def random_walk_BM(N: int, T: float) -> np.ndarray:
    """
    Simula uma trajetória do Movimento Browniano como o limite de um Passeio Aleatório.
    (Começa com passos +/- 1 e aplica a escala correta)
    Recebe como argumentos:
    N -> número de passos (representa a discretização no tempo)
    T -> tempo final do intervalo de simulação

    Retorna um array numpy de tamanho N+1 contendo a trajetória aproximada do MB, com W(0) = 0 e plota seu gráfico.
    """
    passos_simples = np.random.choice([-1, 1], size=N)  # gerando os passos
    # calculando a posição do passeio aleatório em cada instante
    S = np.cumsum(passos_simples)
    # escalando o passeio para convergir para o MB
    W_aprox = np.sqrt(T / N) * S
    return np.insert(W_aprox, 0, 0)  # fazendo W_0 = 0


if __name__ == '__main__':

    # Parâmetros da simulação
    T = 1   # tempo final
    # diferentes números de passos para mostrar a convergência
    num_passos = [10, 50, 300]

    # Configurando o
    plt.figure(figsize=(14, 8))

    num_cores = len(num_passos)
    cmap = plt.get_cmap('YlGnBu')
    cores = cmap(np.linspace(0.3, 1, num_cores))

    # Simulação e plotagem
    for i, N in enumerate(num_passos):
        tempo = np.linspace(0, T, N + 1)
        trajetoria = random_walk_BM(N, T)
        plt.plot(tempo, trajetoria, label=f'N = {N}', color=cores[i])

    # Exportando o gráfico
    #plt.title('Convergência do Passeio Aleatório para o Movimento Browniano')
    plt.xlabel('t', fontsize=20)
    plt.ylabel('$W_t$', fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(title='Núm. Passos (N)', title_fontsize='16', fontsize='14')
    plt.grid(True)

    output_dir = 'Figuras'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    caminho = os.path.join(output_dir, 'passeio_aleatorio.pdf')
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.show()
