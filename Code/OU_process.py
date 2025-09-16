import numpy as np
import matplotlib.pyplot as plt
import os


def OU_simulation(kappa: float, theta: float, sigma: float, X0: float, num_traj: int,
                  T: float = 50, n: int = 500) -> tuple[np.ndarray, np.ndarray]:
    '''Função que simula o processo de Orstein-Uhlenbeck a partir da discretização de Euler-Maruyama.
    Recebe como parâmetros:
    kappa -> Taxa de reversão à média
    theta -> Média a longo prazo
    sigma -> Volatilidade
    X0 -> Valor inicial dos processos
    num_traj -> O número de trajetórias que desejamos simular
    T (opcional) -> Horizonte de tempo. Padrão = 50.
    n (opcional) -> Número de passos de tempo na discretização. Padrão = 500. 

    Retorna um gráfico com as trajetórias simuladas.
    '''

    dt = T/n                                # tamanho de cada passo de tempo
    traj = np.zeros((n + 1, num_traj))      # matriz de trajetórias
    traj[0, :] = X0                         # acrescenta o valor inicial X0

    # Gerando todos os incrementos
    W = np.random.normal(scale=np.sqrt(dt), size=(n, num_traj))

    # Simulação
    for i in range(1, n + 1):
        dX = kappa * (theta - traj[i-1, :]) * dt + sigma * \
            W[i-1, :]            # discretização de EM
        # atualiza o valor da trajetória
        traj[i, :] = traj[i-1, :] + dX

    # Gerando o gráfico
    tempo = np.linspace(0, T, n + 1)    # eixo temporal

    return tempo, traj


def OU_proc_plot(tempo: np.ndarray, traj: np.ndarray, theta: float):
    '''
    Função auxiliar para plotar as trajetórias simuladas de um processo de Ornstein-Uhlenbeck.
    '''

    num_sim = traj.shape[1]

    plt.figure(figsize=(11, 7))

    # Gerando o colormap
    colormap = plt.get_cmap('PuBu')

    # Plota cada trajetória separadamente
    for j in range(num_sim):
        cor = colormap(0.3 + 0.7 * (j / (num_sim - 1)))
        plt.plot(tempo, traj[:, j], color=cor, alpha=0.7, linewidth=1.5)

    # Configurando gráfico
    # desenha a linha da média a longo prazo
    plt.axhline(theta, color='r', linestyle='--',
                label='Média a longo prazo ($\\theta$)')
    plt.title('Ornstein-Uhlenbeck', fontsize=17)
    plt.xlabel('t', fontsize=11)
    plt.ylabel('$X(t)$', fontsize=11)
    plt.grid(True)
    plt.legend()

    output_dir = 'Figuras'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    caminho = os.path.join(output_dir, 'OU_process.pdf')
    plt.savefig(caminho, dpi=300, bbox_inches='tight')
    plt.show()


######################################## Execução de teste ################################################

if __name__ == "__main__":

    # Parâmetros do modelo
    kappa = 0.5
    theta = 10
    sigma = 0.8
    X0 = 25
    num_traj = 5

    # Simulando os dados
    tempo, traj = OU_simulation(kappa, theta, sigma, X0, num_traj)

    # Plotando o resultado
    OU_proc_plot(tempo, traj, theta)
