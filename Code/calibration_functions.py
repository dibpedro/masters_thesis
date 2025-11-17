import numpy as np
import pandas as pd
import os
import heston_pricing as hp
from scipy.optimize import approx_fprime
import time
import BS_equations as bs
from typing import List, Dict, Callable, Tuple, Optional, Any

pasta = r'C:\Pedro\Python\Dissertação\Dados'


def calibration_cleaning(ticker: str, spot: float, r: float = 0.15, min_trades: int = 10,
                         tau_min: int = 30, tau_max: int = 504,
                         min_price: float = 0.01, min_mon: float = 0.75,
                         max_mon: float = 1.25) -> Optional[pd.DataFrame]:
    '''
    Limpa e filtra os dados de opções de um ativo para a calibração.
    Converte opções put em preços de call equivalentes usando a paridade put-call.

    Recebe como parâmetros:
    ticker -> Código do ativo (ex: 'BBAS3')
    spot -> Preço atual do ativo
    r -> Taxa de juros livre de risco (anualizada)
    min_trades -> (Opcional) Número mínimo de negócios
    tau_min -> (Opcional) Tempo mínimo até o vencimento (dias úteis)
    tau_max -> (Opcional) Tempo máximo até o vencimento (dias úteis)
    min_price -> (Opcional) Preço mínimo de mercado da opção
    min_mon -> (Opcional) Moneyness mínimo (S/K)
    max_mon -> (Opcional) Moneyness máximo (S/K)

    Retorna um DataFrame com os dados limpos.

    '''

    arquivo = f'Opções {ticker} - CALLs e PUTs - lista, pesquisa e cotações.xlsx'
    caminho = os.path.join(pasta, arquivo)

    try:
        df = pd.read_excel(caminho, header=1)
    except FileNotFoundError:
        print(f'Arquivo não encontrado - {caminho}')
        return None

    # Mapeando os nomes das colunas
    nomes = {
        'Strike': 'K',
        'Dias úteis': 'tau (dias)',
        'Último': 'V_mkt',
        'Vol. Imp.': 'vol. imp.',
        'Vega': 'vega',
        'Tipo': 'tipo',
        'Núm. de Neg.': 'trades',
        'Mod.': 'mod'
    }

    # Renomeando e selecionando colunas
    colunas = list(nomes.values())
    df = df.rename(columns=nomes)
    df = df[colunas]

    # Ajustando as unidades
    df['K'] = df['K'] / 100
    df['vol. imp.'] = df['vol. imp.'] / 1000
    df['trades'] = pd.to_numeric(df['trades'], errors='coerce')
    df['vega'] = pd.to_numeric(df['vega'], errors='coerce')/10000
    calib_data = df.dropna(subset=colunas)

    # Filtrando apenas por opções europeias
    calib_data = calib_data[calib_data['mod'] == 'E'].copy()

    if calib_data.empty:
        print(f'Nenhuma opção europeia encontrada para {ticker}')
        return None

    # Filtando por liquidez e relevância
    calib_data = calib_data[calib_data['trades'] >= min_trades]
    calib_data = calib_data[(calib_data['tau (dias)'] >= tau_min) & (
        calib_data['tau (dias)'] <= tau_max)]
    calib_data = calib_data[calib_data['V_mkt'] > min_price]

    # Calculando o moneyness (S/K)
    calib_data['moneyness'] = spot / calib_data['K']

    # Condição para CALLs OTM e ATM (S/K <= 1)
    call_cond = (
        (calib_data['tipo'] == 'CALL') &
        (calib_data['moneyness'] >= min_mon) &
        (calib_data['moneyness'] <= 1)
    )

    # Condição para PUTs ITM e ATM (S/K >= 1)
    put_cond = (
        (calib_data['tipo'] == 'PUT') &
        (calib_data['moneyness'] >= 1) &
        (calib_data['moneyness'] <= max_mon)
    )

    # Aplicando os filtros de moneyness
    calib_data = calib_data[call_cond | put_cond].reset_index(drop=True)

    # Convertendo dias úteis para anos
    calib_data['tau (anos)'] = calib_data['tau (dias)'] / 252

    # Paridade Put-Call
    calib_data['equiv_price'] = np.where(
        calib_data['tipo'] == 'CALL',                # Condição: é uma CALL?
        # Se sim: usa o V_mkt original
        calib_data['V_mkt'],
        # Se não (é PUT): calcula C = P + S - K*exp(-r*T)
        calib_data['V_mkt'] + spot - calib_data['K'] *
        np.exp(-r * calib_data['tau (anos)'])
    )

    return calib_data


def v0_estimator(df: Optional[pd.DataFrame], spot: float, v0_def: float = 0.4) -> float:
    '''
    Estima a variância inicial (v0) para o modelo de Heston.
    Utiliza a volatilidade implícita da opção "at-the-money" (K mais próximo
    de S) com o vencimento mais curto disponível.

    Recebe como parâmetros:
    df -> DataFrame de dados de opções (saída de calibration_cleaning)
    spot -> Preço atual do ativo
    v0_def -> (Opcional) Valor padrão para v0 caso a estimativa falhe

    Retorna o valor estimado de v0.
    '''

    if df is None or df.empty:
        print('Dataframe vazio.')
        return v0_def

    try:
        # Encontrando o vencimento mais curto
        min_tau = df['tau (anos)'].min()
        tau_curto = df[df['tau (anos)'] == min_tau]

        if tau_curto.empty:
            print('Nenhuma opção encontrada com a menor maturidade')
            return v0_def

        # Encontrando a opção ATM
        tau_curto = tau_curto.copy()
        tau_curto['dif(K,S)'] = abs(tau_curto['K'] - spot)
        idx_atm = tau_curto['dif(K,S)'].idxmin()

        # Pegando a volatilidade implícita dessa opção
        vol_imp_curto = tau_curto.loc[idx_atm, 'vol. imp.']
        strike_curto = tau_curto.loc[idx_atm, 'K']

        v0 = vol_imp_curto**2
        return v0

    except Exception as e:
        print('Erro inesperado ao estimar v0: {e}')
        print('Utilizando v0 padrão')
        return v0_def


def df_converter(df: Optional[pd.DataFrame], cols: Dict[str, str],
                 min_vega: float = 0.05) -> List[Dict[str, Any]]:
    '''
    Converte o DataFrame de calibração para uma lista de dicionários.
    Calcula os pesos (w_price, w_vol) com base no vega da opção e filtra
    opções com vega muito baixo.

    Recebe como parâmetros:
    df -> DataFrame de dados de opções
    cols -> Dicionário mapeando nomes de colunas do df para nomes esperados
    min_vega -> (Opcional) Valor de vega mínimo para incluir a opção

    Retorna uma lista de dicionários, onde cada dicionário representa uma opção.
    '''

    if df is None or df.empty:
        print('Dataframe vazio')
        return []

    orig_cols = list(cols.keys())
    if not all(col in df.columns for col in orig_cols):
        missing = [col for col in orig_cols if col not in df.columns]
        print('Dataframe com colunas faltando')
        return []

    mkt_data = []
    for index, row in df.iterrows():
        pto = {}
        for orig, new in cols.items():
            pto[new] = row[orig]

        vega = row['vega']
        vol_imp_mkt = row['vol. imp.']

        # Filtrando opções com vega instável
        if vega < min_vega:
            continue

        # Definindo os pesos e adicionando a vol de mercado
        pto['w_price'] = 1/vega
        pto['w_vol'] = vega
        pto['vol_mkt'] = vol_imp_mkt

        mkt_data.append(pto)

    mkt_data_final = [p for p in mkt_data if p.get(
        'w_price', -1) > 0 and p.get('w_vol', -1) > 0]

    if not mkt_data_final:
        print('Lista de dados formatados está vazia')

    return mkt_data_final


def loss_price(spot: float, r: float, v0: float, kappa: float,
               theta: np.ndarray, data_list: list) -> float:
    '''
    Calcula a função de perda da calibração baseada nos preços.
    O erro é a soma ponderada dos quadrados das diferenças entre o preço
    de mercado (V_mkt) e o preço teórico de Heston (V).

    Recebe como parâmetros:
    spot -> Preço atual do ativo
    r -> Taxa de juros livre de risco (anualizada)
    v0 -> Variância inicial (fixa)
    kappa -> Taxa de reversão (fixa)
    theta -> Vetor de parâmetros a otimizar: [v_bar, gamma, rho]
    data_list -> Lista de dicionários (saída de df_converter)

    Retorna o erro quadrático total ponderado.
    '''
    try:
        # Descompactando o vetor de parâmetros
        v_bar = theta[0]
        gamma = theta[1]
        rho = theta[2]
    except IndexError:
        print('Vetor de parâmetros inválido')
        return np.inf

    erro_acumulado = 0

    for opt in data_list:
        V_mkt = opt['V_mkt']
        K = opt['K']
        tau = opt['tau (anos)']
        w = opt['w_price']

        try:
            # Calculando o preço de Heston
            V = hp.heston_call_price(
                S0=spot, K=K, r=r, tau=tau, v0=v0, kappa=kappa, v_bar=v_bar, gamma=gamma, rho=rho)

            if not np.isfinite(V):
                return np.inf
        except Exception as e:
            return np.inf

        erro_opt = w*(V_mkt - V)**2
        erro_acumulado = erro_acumulado + erro_opt

    return erro_acumulado


def loss_vol(spot: float, r: float, v0: float, kappa: float,
             theta: np.ndarray, data_list: list) -> float:
    '''
    Calcula a função de perda da calibração baseada na volatilidade implícita.
    Calcula o preço teórico de Heston (V), encontra sua volatilidade implícita
    (vol) e calcula a soma dos quadrados das diferenças com a vol. de mercado.

    Recebe como parâmetros:
    spot -> Preço atual do ativo
    r -> Taxa de juros livre de risco
    v0 -> Variância inicial (fixa)
    kappa -> Taxa de reversão (fixa)
    theta -> Vetor de parâmetros a otimizar: [v_bar, gamma, rho]
    data_list -> Lista de dicionários (saída de df_converter)

    Retorna o erro quadrático total.
    '''

    try:
        # Descompactando o vetor de parâmetros
        v_bar = theta[0]
        gamma = theta[1]
        rho = theta[2]
    except IndexError:
        print('Vetor de parâmetros inválido')
        return np.inf

    erro_acumulado = 0
    penalidade = 1e10       # Penalidade alta para preços/vols inválidas

    for opt in data_list:

        vol_mkt = opt['vol_mkt']  # Vol implícita de mercado
        K = opt['K']
        tau = opt['tau']
        # w = opt['w_vol']
        w = 1   # Usando RMSE (sem ponderação)

        try:
            # Calculando o preço teórico de Heston
            V = hp.heston_call_price(
                S0=spot, K=K, r=r, tau=tau, v0=v0, kappa=kappa, v_bar=v_bar, gamma=gamma, rho=rho)

            if not np.isfinite(V):
                erro_acumulado = erro_acumulado + penalidade
                continue

            # Invertendo o preço para achar a vol implícita
            vol = bs.vol_imp(V, spot, K, tau, r, vol_mkt)

            if not np.isfinite(vol):
                print(
                    f'Aviso: Vol. imp. inválida ({vol}) para K={K}, tau={tau}. Aplicando penalidade')
                erro_acumulado = erro_acumulado + penalidade
                continue

        except Exception as e:
            print(f'Exceção no cálculo. {e}. Aplicando penalidade.')
            erro_acumulado = erro_acumulado + penalidade
            return np.inf

        erro_opt = w * (vol_mkt - vol)**2
        erro_acumulado = erro_acumulado + erro_opt

    # print(f'Erro para theta {theta}: {erro_acumulado}') # Debug
    return erro_acumulado


def approx_grad(wrap: Callable, spot: float, r: float, v0: float, kappa: float,
                theta: np.ndarray, data_list: list) -> np.ndarray:
    '''
    Calcula o gradiente da função de perda por aproximação numérica (diferenças finitas).
    Utiliza a função approx_fprime da Scipy para estimar o gradiente.

    Recebe como parâmetros:
    wrap -> Função de perda a ser diferenciada (ex: cf.loss_vol)
    spot -> Preço atual do ativo
    r -> Taxa de juros livre de risco
    v0 -> Variância inicial (fixa)
    kappa -> Taxa de reversão (fixa)
    theta -> Ponto onde o gradiente será calculado
    data_list -> Lista de dicionários (saída de df_converter)

    Retorna o vetor gradiente (numpy array).
    '''
    try:
        # Definindo o passo (h) para a diferenciação
        epsilon = np.sqrt(np.finfo(float).eps)  # ~1.49e-8
    except Exception as e:
        print(f'Erro ao calcular epsilon: {e}. Utilizando valor padrão')
        epsilon = 1.49e-8

    # Criando uma função wrapper que aceita apenas 'theta'
    def wrapper(t): return wrap(spot, r, v0, kappa, t, data_list)

    try:
        # Calculando o gradiente [dL/dv_bar, dL/dgamma, dL/drho]
        grad = approx_fprime(theta, wrapper, epsilon)

        if np.any(np.isnan(grad)) or np.any(np.isinf(grad)):
            print(
                f'Gradiente inválido (NaN/inf) calculado para theta = {theta}')
            return np.full_like(theta, np.nan)

        return grad

    except Exception as e:
        print(f'Erro ao aproximarmos o gradiente para theta = {theta}')
        return np.full_like(theta, np.nan)


def proj(theta: np.ndarray) -> np.ndarray:
    '''
    Projeta o vetor de parâmetros 'theta' em um conjunto de restrições (box constraints).
    Garante que os parâmetros [v_bar, gamma, rho] permaneçam dentro
    dos limites definidos.

    Recebe como parâmetros:
    theta -> Vetor de parâmetros [v_bar, gamma, rho]

    Retorna o vetor de parâmetros projetado (numpy array).
    '''

    # Limites [v_bar, gamma, rho]
    lim_inf = np.array([0.08, 0.1, -0.999])
    lim_sup = np.array([5, 7, 0.999])

    # Aplicando os limites (projeção)
    proj_theta = np.clip(theta, lim_inf, lim_sup)

    return proj_theta


def ac_pg(funcs: Dict[str, Callable], otim: Dict[str, Any],
          params: Dict[str, Any]) -> Tuple[Optional[np.ndarray], List[float]]:
    '''
    Executa o algoritmo de Otimização por Gradiente Projetado (PG) ou com busca de linha adaptativa (AC-PG).
    Minimiza a função de perda para encontrar os parâmetros ótimos de Heston.

    Recebe como parâmetros:
    funcs -> Dicionário contendo as funções 'loss', 'grad' e 'proj'
    otim -> Dicionário com parâmetros da otimização:
            'theta0', 'max_iter', 'M0' (passo inicial), 'tol'
    params -> Dicionário com parâmetros do modelo e dados:
             'spot', 'r', 'v0', 'kappa', 'data_list'

    Retorna uma tupla (theta_final, hist_custo):
    - theta_final: O vetor de parâmetros otimizado [v_bar, gamma, rho]
    - hist_custo: Uma lista com o histórico do erro por iteração
    '''

    try:
        # Descompactando as funções
        loss = funcs['loss']
        grad = funcs['grad']
        proj = funcs['proj']

    except KeyError as e:
        print('Chave ausente no dicionário de funções: {e}')
        return None, []

    try:
        # Descompactando parâmetros da otimização
        theta0 = otim['theta0']
        max_iter = otim['max_iter']

        # Definindo o tipo de algoritmo (PG ou AC-PG)
        alpha = otim.get('alpha', None)
        M_hat = 0
        if alpha is None:
            M0 = otim['M0']
            M_hat = M0
            # print('Alg.: AC-PG')

        else:
            print('Alg.: PG')

    except KeyError as e:
        print('Chave ausente no dicionário de parâmetros da otimização: {e}')
        return None, []

    try:
        # Descompactando parâmetros do modelo
        spot = params['spot']
        r = params['r']
        v0 = params['v0']
        kappa = params['kappa']
        data_list = params['data_list']

    except KeyError as e:
        print('Chave ausente no dicionário de parâmetros do modelo: {e}')
        return None, []

    theta = theta0.copy()
    hist_theta = [theta]
    hist_custo = []

    start_time = time.time()

    # Criando wrappers para simplificar as chamadas
    def loss_wrapper(t): return loss(spot, r, v0, kappa, t, data_list)
    def grad_wrapper(t): return grad(loss, spot, r, v0, kappa, t, data_list)

    # Loop de calibração
    for t in range(max_iter):
        theta_prev = theta.copy()

        # Calculando custo e gradiente no ponto atual
        L_prev = loss_wrapper(theta_prev)
        grad_prev = grad_wrapper(theta_prev)

        if np.any(np.isnan(grad_prev)) or not np.isfinite(L_prev):
            print(
                f'Erro na iteração {t+1}: Custo ({L_prev}) ou gradiente ({grad_prev}) é NaN/inf')
            print(f'Parâmetros com falha: {theta_prev}')
            print('Otimização interrompida')
            return theta_prev, hist_custo

        hist_custo.append(L_prev)

        label = 0

        # Definindo o tamanho do passo (alpha_t)
        if alpha is not None:
            alpha_t = alpha
            label = alpha_t

        else:
            alpha_t = 1/M_hat
            label = M_hat

        z_t = theta_prev - alpha_t*grad_prev    # Passo
        theta = proj(z_t)   # Projeção

        # Condição de Feller
        v_bar = theta[0]
        gamma = theta[1]

        if gamma**2 > 2*kappa*v_bar:
            theta[1] = np.sqrt(2*kappa*v_bar)

        L_curr = loss_wrapper(theta)
        diff_theta = theta - theta_prev

        # Atualização do M_hat
        if alpha is None:
            grad_term = np.dot(grad_prev, diff_theta)
            num = 2*(L_curr - L_prev - grad_term)
            den = np.linalg.norm(diff_theta)**2

            # Evitando divisão por zero
            if den > 1e-12:
                M_t = num/den
                M_hat = max(M_hat, M_t)

        # if (t + 1) % 50 == 0 or t == 0:
            # print(f'{t+1:<7} | {L_prev:<17.6f} | {M_hat:<17.6f} | {alpha_t:<17.6f}')

        # Critério de parada
        if np.linalg.norm(diff_theta) < otim.get('tol', 1e-6):
            # print(f'Convergência atingida na iteração {t+1} (variação < {otim.get('tol', 1e-6)})')
            break

    # if t == max_iter - 1:
        # print('Máximo de iterações atingido')

    end_time = time.time()
    # print(f'Otimização concluída em {end_time - start_time:.2f} segundos')

    try:
        # Adicionando o custo final ao histórico
        final_cost = loss_wrapper(theta)
        hist_custo.append(final_cost)
        # print(f'Custo final: {final_cost:.6f}')

    except Exception as e:
        print(f'Erro ao calcular custo final: {e}')
        pass

    return theta, hist_custo
