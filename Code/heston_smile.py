import numpy as np
import matplotlib.pyplot as plt
from heston_pricing import heston_call_price
from BS_equations import vol_imp


def heston_smile(S0: float, tau: float, r: float, v0: float, heston_params: dict, strikes: np.array) -> np.array:
    '''
    Calcula os dados da curva smile no modelo de Heston.
    Recebe como parâmetros:

    Retorna um array com as vol. imp. correspondentes a strike.
    '''

    # Lista para os resultados
    vol_imp_list = []
    sigma_inicial = np.sqrt(v0)     # Nosso chute inicial para a vol. imp.

    for k in strikes:
        # Obtendo o preço de Heston
        price = heston_call_price(S0, k, v0, r, tau, **heston_params)

        # Calculando a vol. imp.
        # Usando price como C_mkt
        vol = vol_imp(price, S0, k, tau, r, sigma_inicial)

        # Salvando na lista
        vol_imp_list.append(vol)

    return np.array(vol_imp_list)
