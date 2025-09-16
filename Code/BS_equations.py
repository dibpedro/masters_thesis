import numpy as np
from scipy.stats import norm


def BS_call_price(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    '''Implementa a fórmula do valor teórico para o preço de uma call europeia no modelo Black-Scholes.
    Recebe como parâmetros:
    S = preço atual do ativo
    K = preço de exercício
    tau = tempo até o vencimento
    r = taxa de juros livre de risco (anualizada)
    sigma = volatilidade (anualizada)

    Retorna o preço da call.
    '''

    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * tau)/(sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)

    N_d1 = norm.cdf(d1)
    N_d2 = norm.cdf(d2)

    call_price = (S*N_d1 - K*np.exp(-r * tau) * N_d2)

    return call_price


def BS_vega(S: float, K: float, tau: float, r: float, sigma: float) -> float:
    '''Implementa a fórmula do valor teórico para o vega de uma call europeia no modelo Black-Scholes.
    Recebe como parâmetros:
    S = preço atual do ativo
    K = preço de exercício
    tau = tempo até o vencimento
    r = taxa de juros livre de risco (anualizada)
    sigma = volatilidade (anualizada)

    Retorna o valor anualizado do vega.
    '''

    d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * tau)/(sigma * np.sqrt(tau))

    n_d1 = norm.pdf(d1)

    vega = (S * np.sqrt(tau) * n_d1)

    return vega


def vol_imp(C_mkt: float, S: float, K: float, tau: float, r: float, sigma_inicial: float) -> float:
    '''Calcula a voltilidade implícita de uma opção de compra europeia utilizando o algoritmo de Newton-Raphson.
    Recebe como parâmetros:
    C_mkt = o valor de mercado da call
    S = preço atual do ativo
    K = preço de exercício
    tau = tempo até o vencimento
    r = taxa de juros livre de risco (anualizada)
    sigma_inicial = chute inicial para a volatilidade (anualizada)

    Faz uso das funções BS_call_price() e BS_vega() previamente definidas.
    Retorna o valor da volatilidade implícita. 
    '''

    erro = 1e10

    while erro > 10e-10:
        g = C_mkt - BS_call_price(S, K, tau, r, sigma_inicial)
        vega = -BS_vega(S, K, tau, r, sigma_inicial)
        sigma_imp = sigma_inicial - g/vega

        erro = abs(sigma_imp - sigma_inicial)
        sigma_inicial = sigma_imp

    return sigma_imp


################################################################### Demonstração ##################################################################

if __name__ == '__main__':

    # Parâmetros de exemplo
    S = 100
    K = 105
    tau = 1
    r = 0.05
    sigma = 0.5

    # Calculando o preço da call e o vega:
    price = BS_call_price(S, K, tau, r, sigma)
    print(f'O preço teórico da call é {round(price, 2)}')

    vega = BS_vega(S, K, tau, r, sigma)
    print(f'E seu vega é {round(vega, 2)}')
