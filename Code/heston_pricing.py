import numpy as np
from scipy.integrate import quad


def heston_char(k: float, S0: float, K: float, r: float, tau: float, v0: float, kappa: float, v_bar: float, gamma: float, rho: float, j: int):
    '''
    Calcula a função característica do modelo de Heston para as probabilidades P0 e P1 na expressão do preço de uma call europeia.
    A f.c. é dada por exp{Cj(k, tau)v_bar + Dj(k, tau)v + ikx}.
    Recebe como parâmetros:
    k -> Variável de integração de Fourier
    S0 -> Preço inicial do ativo
    K -> Preço de exercício
    v0 -> Variância inicial 
    r -> Taxa de juros livre de risco
    tau -> Maturidade
    kappa -> Taxa de reversão à média
    v_barra -> Média a longo prazo
    gamma -> Volatilidade da volatilidade
    rho -> Coeficiente de correlação
    j -> Índice da probabilidade (1 para P1, 0 para P0)

    Retorna o valor complexo do termo exponencial.
    '''

    i = 1j  # unidade imaginária

    # Definindo os coeficientes:
    if j == 1:
        bj = kappa - rho*gamma
    elif j == 0:
        bj = kappa
    else:
        raise ValueError('O índice deve ser 0 ou 1')

    alpha_j = (-k**2)/2 - i*k*(0.5 - j)
    beta_j = bj - i*k*rho*gamma
    eta = (gamma**2)/2

    # Solução da eq de Riccati
    dj = np.sqrt((bj - i*k*rho*gamma)**2 + (k**2 - 2*i*k*(j - 0.5))*gamma**2)
    r_plus = (beta_j + np.sqrt(beta_j**2 - 4*alpha_j*eta))/(2*eta)
    r_minus = (beta_j - np.sqrt(beta_j**2 - 4*alpha_j*eta))/(2*eta)
    gj = r_minus/r_plus

    Dj = r_minus*((1 - np.exp(-dj*tau))/(1 - gj*np.exp(-dj*tau)))
    Cj = (kappa/gamma**2)*(r_minus*tau*gamma**2 - 2 *
                           np.log((1 - gj*np.exp(-dj*tau))/(1 - gj)))

    # Obtendo a exponencial
    x = np.log(S0/K)  # log-moneyness
    expo = np.exp(Cj*v_bar + Dj*v0 + i*k*x)

    return expo


def heston_integ(k: float, S0: float, K: float, r: float, tau: float, v0: float, kappa: float, v_bar: float, gamma: float, rho: float, j: int):
    '''
    Calcula o integrando necessário para obtermos os valores de P0 e P1.
    Recebe como argumento os mesmo parâmetros de heston_char().

    Retorna a parte real da função.
    '''
    # Obtendo o integrando
    i = 1j
    expo = heston_char(k, S0, K, r, tau, v0, kappa, v_bar, gamma, rho, j)

    return (expo/(i*k)).real


def heston_prob(S0: float, K: float, r: float, tau: float, v0: float, kappa: float, v_bar: float, gamma: float, rho: float, j: int):
    '''
    Calcula as probabilidades P0 e P1 para preço da call no modelo de Heston. Utiliza a função heston_char() para calcular a função
    característica, heston_integ() para fornecer um integrando e utiliza o método da quadratura adaptativa para calcular a integral da parte real.
    Recebe como parâmetros:
    S0 -> Preço inicial do ativo
    K -> Preço de exercício     v0 -> Variância inicial 
    r -> Taxa de juros livre de risco
    tau -> Maturidade
    kappa -> Taxa de reversão à média
    v_barra -> Média a longo prazo
    gamma -> Volatilidade da volatilidade
    rho -> Coeficiente de correlação
    j -> Índice da probabilidade (1 para P1, 0 para P0)

    Retorna o valor da probabilidade procurada.
    '''
    # Realizando a integração numérica
    integral, _ = quad(heston_integ, 0, np.inf, args=(
        S0, K, r, tau, v0, kappa, v_bar, gamma, rho, j), limit=500, epsabs=1e-10, epsrel=1e-10)

    # Obtendo a probabilidade
    prob = 0.5 + (1/np.pi)*integral

    return prob


def heston_call_price(S0: float, K: float, r: float, tau: float, v0: float, kappa: float, v_bar: float, gamma: float, rho: float):
    '''
    Calcula o preço de uma call europeia no modelo de Heston a partir da expressão C(K) = S*P1 - Ke^{-rtau}P0.
    Chama as funções heston_char() e heston_prob() para calcular os valores necessários.
    Recebe os mesmos parâmetros de heston_prob(), com exceção de j.

    Retorna o preço justo para a call.
    '''

    # Calculando as probabilidades
    P0 = heston_prob(S0, K, r, tau, v0, kappa, v_bar, gamma, rho, j=0)
    P1 = heston_prob(S0, K, r, tau, v0, kappa, v_bar, gamma, rho, j=1)

    # Fórmula do preço
    C = S0*P1 - K*np.exp(-r * tau)*P0

    return C


###################################################### Execução de teste #########################################################################
if __name__ == "__main__":

    # Parâmetro do exemplo
    S0 = 100
    K = 100
    tau = 1
    r = 0.05
    v0 = 0.04

    # Parâmetros do modelo
    kappa = 2
    v_bar = 0.05
    gamma = 0.6
    rho = -0.7

    resultado = heston_call_price(S0, K, r, tau, v0, kappa, v_bar, gamma, rho)
    print(resultado)
