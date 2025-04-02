import talib as ta
import pandas as pd
import numpy as np
import logging
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller


def feature_engineering_pipeline(df, symbol, market_returns=None):
    """
    Pipeline integrado para adicionar uma série de indicadores financeiros e características (features)
    ao DataFrame de dados de mercado.

    Esta função orquestra o processo de engenharia de features, realizando cálculos de indicadores técnicos,
    métricas financeiras e recursos adicionais para análise preditiva e modelagem.

    Parâmetros:
        df: DataFrame contendo os dados de mercado, como preços e volumes.
        symbol: O símbolo do ativo financeiro (por exemplo, 'AAPL' para Apple).
        market_returns: Retornos logarítmicos do mercado (opcional), usado para calcular métricas como Beta e correlação.

    Retorna:
        df: DataFrame com novas colunas de indicadores e métricas, sem valores ausentes (dropna).
    """
    # 1. Validação dos dados de entrada
    df = validate_inputs(df)
    # A função `validate_inputs` valida a integridade dos dados e garante que as colunas necessárias estejam presentes no DataFrame.

    # 2. Cálculo dos indicadores principais (como SMA, EMA, MACD, etc.)
    df = calcular_indicadores(df, symbol)
    # A função `calcular_indicadores` adiciona indicadores técnicos como médias móveis, RSI, MACD, etc., ao DataFrame.

    # 3. Adição dos indicadores Ichimoku Cloud
    df = add_ichimoku_cloud(df)
    # A função `add_ichimoku_cloud` adiciona os componentes do indicador Ichimoku, que ajuda a detectar tendências e suportes/resistências.

    # 4. Adição do indicador VW-MACD (MACD ponderado por volume)
    df = add_vwmacd(df)
    # A função `add_vwmacd` calcula o VW-MACD, uma versão do MACD que leva em consideração o volume de negociação.

    # 5. Adição de estatísticas avançadas (como Exponente de Hurst, assimetria e curtose do volume)
    df = add_advanced_statistics(df)
    # A função `add_advanced_statistics` adiciona métricas como assimetria, curtose e o Exponente de Hurst, que ajudam a entender a distribuição do volume.

    # 6. Adição de métricas de risco (como drawdown, VaR, etc.)
    df = add_risk_metrics(df)
    # A função `add_risk_metrics` calcula métricas de risco como drawdown e Value at Risk (VaR) para avaliar a exposição ao risco do ativo.

    # 7. Adição de features de aprendizado de máquina (como lags e outras variáveis temporais)
    df = add_ml_features(df)
    # A função `add_ml_features` adiciona variáveis temporais, como lags, para ajudar a capturar dependências temporais no modelo de aprendizado de máquina.

    # 8. Adição de métricas financeiras adicionais (como Sharpe ratio, Sortino ratio, etc.)
    df = add_additional_metrics(df, market_returns)
    # A função `add_additional_metrics` calcula métricas como Sharpe ratio, Sortino ratio e outras, com base nos retornos do ativo e do mercado.

    # 9. Remoção de valores ausentes
    return df.dropna()
    # A função `dropna()` remove qualquer linha que contenha valores ausentes (NaN) no DataFrame final. Isso garante que o modelo ou análise subsequente não seja afetado por dados faltantes.


# Função para calcular indicadores principais do mercado
def calcular_indicadores(df, symbol):
    """
    Calcula vários indicadores técnicos e adiciona as colunas ao DataFrame.

    Parâmetros:
        df: DataFrame contendo os dados de preços e volume.
        symbol: Símbolo do ativo financeiro para coletar o preço de bid e ask.

    Retorna:
        df: DataFrame com novos indicadores calculados.
    """
    try:
        # Adiciona preços de bid e ask, além de preço médio
        # O símbolo é usado para coletar as informações de bid e ask através do MetaTrader 5 (mt5)
        df['bid'] = [mt5.symbol_info_tick(symbol).bid for _ in range(len(df))]
        df['ask'] = [mt5.symbol_info_tick(symbol).ask for _ in range(len(df))]
        df['last_price'] = (df['bid'] + df['ask']) / 2  # Preço médio (entre bid e ask)

        # Calcula a volatilidade (diferença entre o preço máximo e o preço mínimo)
        df['volatility'] = df['high'] - df['low']

        # Calcula a mudança percentual do preço entre o fechamento e a abertura
        df['change_percent'] = ((df['close'] - df['open']) / df['open']) * 100

        # Calcula o preço médio como a média simples dos quatro preços (abertura, alta, baixa, fechamento)
        df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # Calculando médias móveis simples (SMA) e exponenciais (EMA) para diferentes períodos
        df = calculate_moving_average(df, 'close', [5, 9, 21, 50, 100, 200], ma_type="SMA")
        df = calculate_moving_average(df, 'close', [5, 9, 21, 50, 100, 200], ma_type="EMA")

        # Calcula o Índice de Força Relativa (RSI) de 14 e 9 períodos
        df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)
        df['rsi_9'] = ta.RSI(df['close'], timeperiod=9)

        # Calcula o MACD e a linha de sinal
        macd, signal_macd, _ = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['signal_macd'] = signal_macd

        # Calcula as Bandas de Bollinger para diferentes períodos (21, 9, 50)
        upper, middle, lower = ta.BBANDS(df['close'], timeperiod=21)
        df['bb_upper21'] = upper
        df['bb_lower21'] = lower

        upper, middle, lower = ta.BBANDS(df['close'], timeperiod=9)
        df['bb_upper9'] = upper
        df['bb_lower9'] = lower

        upper, middle, lower = ta.BBANDS(df['close'], timeperiod=50)
        df['bb_upper50'] = upper
        df['bb_lower50'] = lower

        # Novos Indicadores Técnicos:
        # Average True Range (ATR) para 14 períodos
        df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        # Rate of Change (ROC) para 10 períodos
        df['roc_10'] = ta.ROC(df['close'], timeperiod=10)
        # On-Balance Volume (OBV) para volume ajustado
        df['obv'] = ta.OBV(df['close'], df['tick_volume'])
        # Money Flow Index (MFI) para 14 períodos
        df['mfi_14'] = ta.MFI(df['high'], df['low'], df['close'], df['tick_volume'], timeperiod=14)

        # Estatísticas de volatilidade e momentum
        # Rolling max, min e desvio padrão para os últimos 20 períodos
        df['rolling_max_20'] = df['close'].rolling(window=20).max()
        df['rolling_min_20'] = df['close'].rolling(window=20).min()
        df['std_dev_20'] = df['close'].rolling(window=20).std()

        # ADX (Average Directional Index) - Força da Tendência
        df['adx_14'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # VWAP (Volume Weighted Average Price) - Preço Médio Ponderado por Volume
        df['vwap'] = (df['close'] * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()

        # Médias móveis do OBV para 10, 20, 50, e 100 períodos
        df['obv_10'] = df['obv'].rolling(window=10).mean()
        df['obv_20'] = df['obv'].rolling(window=20).mean()
        df['obv_50'] = df['obv'].rolling(window=50).mean()
        df['obv_100'] = df['obv'].rolling(window=100).mean()

    except Exception as e:
        # Se ocorrer um erro, ele será registrado no log de exceções
        logging.exception("Erro em calcular_indicadores: %s", e)

    # Retorna o DataFrame com as novas colunas de indicadores calculados
    return df


def add_ichimoku_cloud(df):
    """
    Adiciona os indicadores da nuvem Ichimoku ao DataFrame.

    A Nuvem Ichimoku é composta por cinco componentes principais:
    1. Conversion Line (Tenkan-sen) - 'ichi_conversion'
    2. Base Line (Kijun-sen) - 'ichi_base'
    3. Leading Span A (Senkou Span A) - 'ichi_leading_span_a'
    4. Leading Span B (Senkou Span B) - 'ichi_leading_span_b'

    A função calcula essas linhas e as adiciona ao DataFrame.

    Parâmetros:
        df: DataFrame contendo os dados de preços (high e low).

    Retorna:
        df: DataFrame com as novas colunas de indicadores Ichimoku.
    """
    # Linha de conversão (Tenkan-sen): é a média do maior máximo e menor mínimo de 9 períodos.
    df['ichi_conversion'] = (df['high'].rolling(9).max() + df['low'].rolling(9).min()) / 2

    # Linha base (Kijun-sen): é a média do maior máximo e menor mínimo de 26 períodos.
    df['ichi_base'] = (df['high'].rolling(26).max() + df['low'].rolling(26).min()) / 2

    # Leading Span A (Senkou Span A): é a média das linhas de conversão e base, deslocada para frente por 26 períodos.
    df['ichi_leading_span_a'] = (df['ichi_conversion'] + df['ichi_base']) / 2

    # Leading Span B (Senkou Span B): é a média do maior máximo e menor mínimo de 52 períodos, deslocada para frente por 26 períodos.
    df['ichi_leading_span_b'] = (df['high'].rolling(52).max() + df['low'].rolling(52).min()) / 2

    # Retorna o DataFrame com as novas colunas de indicadores Ichimoku calculados.
    return df


def add_vwmacd(df):
    """
    Adiciona o indicador VW-MACD (Volume Weighted MACD) ao DataFrame.

    O VW-MACD é uma versão modificada do MACD (Moving Average Convergence Divergence)
    que utiliza o preço ponderado pelo volume (VW) em vez do preço simples de fechamento.

    Parâmetros:
        df: DataFrame contendo os dados de preço (close) e volume (tick_volume).

    Retorna:
        df: DataFrame com as novas colunas do VW-MACD calculadas.
    """
    # Calcula o preço ponderado pelo volume (VW) utilizando a fórmula do VWAP (Volume Weighted Average Price)
    # Fazemos a soma cumulativa do preço ponderado pelo volume, dividido pela soma cumulativa do volume
    vw_price = (df['close'] * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()

    # Aplica o MACD no preço ponderado pelo volume (vw_price)
    # O MACD é calculado com um período rápido de 12, um período lento de 26, e um período de sinal de 9
    df['vwmacd'], df['vwmacd_signal'], _ = ta.MACD(vw_price, fastperiod=12, slowperiod=26, signalperiod=9)

    # Retorna o DataFrame com as novas colunas do VW-MACD
    return df

def add_market_regime(df):
    """
    Adiciona indicadores de regime de mercado (força da tendência e regime de volatilidade) ao DataFrame.

    O indicador de regime de mercado avalia a força da tendência (ADX) e se o mercado está em um regime de alta ou baixa volatilidade.

    Parâmetros:
        df: DataFrame contendo os dados de preços (high, low, close) e volatilidade.

    Retorna:
        df: DataFrame com as novas colunas de regime de mercado calculadas.
    """
    # Calcula a força da tendência utilizando o indicador ADX (Average Directional Index)
    # O ADX é calculado com um período de 14 e avalia a força da tendência no mercado
    df['trend_strength'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)

    # Regime de volatilidade: Verifica se a volatilidade atual é maior que a média móvel de 50 períodos da volatilidade
    # Se for maior, atribui 1 (indica regime de alta volatilidade), caso contrário, -1 (indica regime de baixa volatilidade)
    df['volatility_regime'] = np.where(df['volatility'] > df['volatility'].rolling(50).mean(), 1, -1)

    # Retorna o DataFrame com os novos indicadores de regime de mercado
    return df


# Função para adicionar características básicas ao DataFrame
def add_basic_features(df):
    """
    Adiciona características básicas como volatilidade, mudança percentual,
    e preço médio ao DataFrame.

    Parâmetros:
        df: DataFrame contendo os dados de preços.

    Retorna:
        df: DataFrame com características básicas calculadas.
    """
    try:
        # Cálculo de volatilidade
        df['volatility'] = df['high'] - df['low']

        # Cálculo de mudança percentual
        df['change_percent'] = ((df['close'] - df['open']) / df['open']) * 100

        # Preço médio
        df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # Retorno logarítmico
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        # Z-score de volatilidade
        df['zscore_vol'] = df['volatility'].rolling(window=50).apply(
            lambda x: (x[-1] - x.mean()) / x.std(), raw=True)

    except Exception as e:
        logging.exception("Erro em add_basic_features: %s", e)
    return df

def add_advanced_statistics(df):
    """
    Adiciona estatísticas avançadas ao DataFrame, incluindo assimetria (skew), curtose (kurtosis)
    e o cálculo dos fractais de Hurst para identificar a persistência de tendência.

    Parâmetros:
        df: DataFrame contendo os dados de volume e preço (tick_volume, close).

    Retorna:
        df: DataFrame com as novas colunas de estatísticas avançadas calculadas.
    """
    # Assimetria (Skewness) do volume: A assimetria mede a simetria da distribuição de volume.
    # Um valor positivo indica que a distribuição tem uma cauda à direita (tendência para volumes mais altos),
    # enquanto um valor negativo indica uma cauda à esquerda (tendência para volumes mais baixos).
    df['volume_skew'] = df['tick_volume'].rolling(30).skew()

    # Curtose do volume: A curtose mede a "altura" das caudas de uma distribuição.
    # Valores altos indicam distribuições com caudas mais "pesadas" (outliers), e valores baixos indicam distribuições mais "suaves".
    df['volume_kurtosis'] = df['tick_volume'].rolling(30).kurt()

    # Função interna para calcular o Hurst Exponent, utilizado para medir a persistência de tendência
    # O Exponente de Hurst ajuda a identificar se o mercado segue uma tendência (valores próximos de 0.5),
    # é aleatório (valores próximos de 0.5) ou possui tendência persistente (valores maiores que 0.5).
    def hurst(series):
        # A função hurst recebe uma série e calcula o Exponente de Hurst utilizando a técnica de "R/S" (Rescaled Range)
        lags = np.arange(2, 20)  # Lags de 2 a 19 períodos
        tau = [np.std(np.subtract(series[lag:], series[:-lag])) for lag in lags]  # Calcula a variância para cada lag
        poly = np.polyfit(np.log(lags), np.log(tau), 1)  # Ajusta uma linha reta (log-log) para os resultados
        return poly[0] * 2  # O coeficiente da linha ajustada (exponente de Hurst) é multiplicado por 2

    # Aplica a função de Hurst para uma janela de 30 períodos sobre os preços de fechamento.
    df['hurst_30'] = df['close'].rolling(30).apply(hurst)

    # Retorna o DataFrame com as novas colunas de estatísticas avançadas calculadas.
    return df


# Função para calcular a Hull Moving Average (HMA)
def hull_ma(series, window):
    """
    Calcula a Hull Moving Average (HMA) para uma série temporal de preços.

    Parâmetros:
        series: Série temporal de preços.
        window: Tamanho da janela de cálculo para a média.

    Retorna:
        hma_series: Série temporal com a Hull Moving Average.
    """
    wma_half = ta.WMA(series, window // 2)  # Média móvel ponderada de metade do período
    wma_full = ta.WMA(series, window)  # Média móvel ponderada do período completo
    hma_series = ta.WMA(2 * wma_half - wma_full, int(np.sqrt(window)))  # Resultado da HMA
    return hma_series


def rolling_adf(x):
    """
    Aplica o teste ADF (Augmented Dickey-Fuller) para verificar a estacionaridade de uma série temporal.

    O teste ADF é utilizado para determinar se uma série temporal possui uma raiz unitária e, portanto,
    é estacionária ou não. Se a série for estacionária, a hipótese nula será rejeitada.

    Parâmetros:
        x: Série temporal (pode ser qualquer coluna do DataFrame que se deseja testar).

    Retorna:
        result[0]: Estatística de teste ADF. Um valor negativo indica que a série é estacionária.
    """
    # Aplica o teste ADF à série temporal
    result = adfuller(x)

    # Retorna a estatística de teste (primeiro valor do resultado)
    return result[0]  # Retorna a estatística de teste ADF


def add_ml_features(df):
    """
    Adiciona recursos de machine learning ao DataFrame, como clusterização de volatilidade
    e características de wavelets (Ondaletas).

    A clusterização de volatilidade divide os dados em dois grupos: alta e baixa volatilidade,
    e as características de Ondaletas extraem informações de decomposição de sinais com base em Wavelets.

    Parâmetros:
        df: DataFrame contendo os dados de volatilidade e preço (close).

    Retorna:
        df: DataFrame com as novas características de machine learning adicionadas.
    """
    # Clusterização de Volatilidade: classifica a volatilidade em alta (1) ou baixa (0) com base
    # no valor atual de volatilidade comparado à média exponencial (EWMA) de 30 períodos.
    df['vol_cluster'] = (df['volatility'] > df['volatility'].ewm(span=30).mean()).astype(int)

    # Características de Ondaletas (Wavelets): usa a decomposição discreta de wavelet (DWT) para
    # extrair coeficientes de aproximação e detalhamento. Aqui, usamos o "Haar" como a função base.
    import pywt
    coeffs = pywt.dwt(df['close'].tail(30), 'haar')  # Decomposição de 30 períodos usando Haar wavelet

    # Concatenamos os coeficientes de aproximação (coeffs[0]) com zeros para ter o mesmo comprimento
    # do DataFrame original (assumindo que o DataFrame tem mais de 30 linhas).
    df['wavelet_coeff'] = np.concatenate([coeffs[0], np.zeros(len(df)-len(coeffs[0]))])

    # Retorna o DataFrame com as novas colunas adicionadas
    return df


def add_risk_metrics(df):
    """
    Adiciona métricas de risco ao DataFrame, como Expected Shortfall (ES) e Omega Ratio.

    - Expected Shortfall (ES) calcula a média dos retornos mais negativos de um conjunto de dados,
    o que é uma métrica de risco frequentemente usada em finanças.

    - Omega Ratio é uma medida de risco-retorno que avalia o potencial de ganhos versus perdas.

    Parâmetros:
        df: DataFrame contendo os dados de retornos logarítmicos (log_ret).

    Retorna:
        df: DataFrame com as novas métricas de risco adicionadas.
    """
    # Expected Shortfall (ES): Calcula a média dos retornos abaixo do percentil 5 (5% mais negativos).
    df['es_95'] = df['log_ret'].rolling(100).apply(lambda x: x[x <= x.quantile(0.05)].mean())

    # Omega Ratio: A métrica é calculada como a soma dos retornos positivos dividida pela soma dos retornos negativos.
    df['omega_ratio'] = (df['log_ret'].rolling(30).apply(lambda x: x[x > 0].sum()) /
                         -df['log_ret'].rolling(30).apply(lambda x: x[x < 0].sum()))

    # Retorna o DataFrame com as novas métricas de risco adicionadas.
    return df


def add_moving_averages(df):
    """
    Adiciona médias móveis simples (SMA) ao DataFrame, com períodos fixos de 9, 21, 50, 100 e 200.

    As médias móveis são utilizadas para suavizar os dados de preço e identificar tendências,
    sendo muito usadas em análise técnica para indicar se o mercado está em tendência de alta ou baixa.

    Parâmetros:
        df: DataFrame contendo os dados de preços (close).

    Retorna:
        df: DataFrame com as novas colunas de médias móveis adicionadas.
    """
    try:
        # Calculando as médias móveis simples (SMA) para vários períodos
        df['sma_9'] = ta.SMA(df['close'], timeperiod=9)    # SMA de 9 períodos
        df['sma_21'] = ta.SMA(df['close'], timeperiod=21)  # SMA de 21 períodos
        df['sma_50'] = ta.SMA(df['close'], timeperiod=50)  # SMA de 50 períodos
        df['sma_100'] = ta.SMA(df['close'], timeperiod=100) # SMA de 100 períodos
        df['sma_200'] = ta.SMA(df['close'], timeperiod=200) # SMA de 200 períodos

    except Exception as e:
        # Se ocorrer um erro, ele será registrado no log de exceções
        logging.exception("Erro em add_moving_averages: %s", e)

    # Retorna o DataFrame com as novas colunas de médias móveis
    return df


def add_dynamic_indicators(df, config={'sma_periods': [9,21,50], 'rsi_period': 14}):
    """
    Adiciona indicadores dinâmicos ao DataFrame, incluindo médias móveis (SMA) e o Índice de Força Relativa (RSI).

    A função permite a personalização dos períodos das médias móveis e do RSI por meio do parâmetro 'config'.

    Parâmetros:
        df: DataFrame contendo os dados de preços (close).
        config: Dicionário contendo os períodos das médias móveis e do RSI. O padrão é SMA [9, 21, 50] e RSI com 14 períodos.

    Retorna:
        df: DataFrame com os novos indicadores adicionados.
    """
    # Para cada período especificado em 'sma_periods', calcula e adiciona a média móvel simples (SMA)
    for period in config['sma_periods']:
        df[f'sma_{period}'] = ta.SMA(df['close'], period)

    # Calcula o Índice de Força Relativa (RSI) para o período especificado
    df[f'rsi_{config["rsi_period"]}'] = ta.RSI(df['close'], config['rsi_period'])

    # Retorna o DataFrame com as novas colunas de indicadores adicionadas
    return df


def add_volatility_indicators(df):
    """
    Adiciona indicadores de volatilidade ao DataFrame, como ATR, Bandas de Bollinger,
    Keltner Channels e volatilidade Rolling.

    Esses indicadores são amplamente utilizados na análise técnica para avaliar
    a magnitude das flutuações de preço e identificar possíveis condições de sobrecompra ou sobrevenda.

    Parâmetros:
        df: DataFrame contendo os dados de preço (high, low, close).

    Retorna:
        df: DataFrame com as novas colunas de indicadores de volatilidade adicionadas.
    """
    try:
        # 7. ATR de 14 e 50 períodos (Average True Range)
        # ATR é uma medida de volatilidade que considera os intervalos de preço, ajudando a entender
        # a magnitude da variação do preço em um determinado período.
        df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_50'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=50)

        # 8. Volatilidade Rolling de 20 períodos
        # Calcula o desvio padrão dos preços de fechamento nos últimos 20 períodos,
        # o que serve como uma medida de volatilidade histórica.
        df['rolling_volatility_20'] = df['close'].rolling(window=20).std()

        # 9. Bandas de Bollinger
        # As Bandas de Bollinger consistem em uma média móvel simples (SMA) com dois desvios padrão
        # acima e abaixo dela, formando uma "faixa" que ajuda a identificar níveis de sobrecompra ou sobrevenda.
        upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_width'] = upper - lower  # A largura das Bandas de Bollinger indica a volatilidade.
        df['bollinger_percent_b'] = (df['close'] - lower) / (upper - lower)  # Percentual de onde o preço está dentro da banda

        # 10. Keltner Channels
        # Os Keltner Channels são compostos por uma média exponencial (EMA) com uma banda superior e inferior
        # baseada no ATR. A banda superior é a EMA + 2 * ATR, e a banda inferior é a EMA - 2 * ATR.
        # O "keltner_break" indica se o preço ultrapassou essas bandas.
        df['ema_20'] = ta.EMA(df['close'], timeperiod=20)  # EMA de 20 períodos
        df['keltner_upper'] = df['ema_20'] + 2 * df['atr_14']  # Banda superior de Keltner
        df['keltner_lower'] = df['ema_20'] - 2 * df['atr_14']  # Banda inferior de Keltner
        # "keltner_break" indica se o preço ultrapassou a banda superior (+1) ou a inferior (-1),
        # ou se está entre as duas bandas (0).
        df['keltner_break'] = np.where(df['close'] > df['keltner_upper'], 1,
                                       np.where(df['close'] < df['keltner_lower'], -1, 0))

    except Exception as e:
        # Caso ocorra um erro, ele será registrado no log de exceções
        logging.exception("Erro em add_volatility_indicators: %s", e)

    # Retorna o DataFrame com as novas colunas de indicadores de volatilidade
    return df


def add_momentum_indicators(df):
    """
    Adiciona indicadores de momentum ao DataFrame, incluindo o Índice de Força Relativa (RSI),
    o MACD, os osciladores estocásticos e outros indicadores de momentum.

    Indicadores de momentum são utilizados para medir a força de uma tendência e a velocidade
    das mudanças de preço, ajudando a identificar condições de sobrecompra ou sobrevenda.

    Parâmetros:
        df: DataFrame contendo os dados de preço (high, low, close).

    Retorna:
        df: DataFrame com as novas colunas de indicadores de momentum adicionadas.
    """
    try:
        # 11. Indicadores de momentum
        # ROC (Rate of Change) para 5 e 20 períodos
        # ROC calcula a variação percentual de preço entre dois períodos e é usado para avaliar o momentum.
        df['roc_5'] = ta.ROC(df['close'], timeperiod=5)
        df['roc_20'] = ta.ROC(df['close'], timeperiod=20)

        # Williams %R de 14 períodos
        # O Williams %R mede os níveis de sobrecompra ou sobrevenda e a força da tendência de preço.
        df['williams_r_14'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14)

        # Momentum de 10 períodos
        # O indicador de momentum mede a velocidade da mudança no preço do ativo.
        df['momentum_10'] = ta.MOM(df['close'], timeperiod=10)

        # RSI de 14 períodos (Índice de Força Relativa)
        # O RSI é utilizado para identificar condições de sobrecompra ou sobrevenda em um ativo.
        df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)

        # 12. MACD (Moving Average Convergence Divergence)
        # O MACD é um indicador que mede a convergência e divergência das médias móveis para identificar mudanças na tendência.
        macd, macd_signal, macd_hist = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist  # Histograma do MACD

        # 13. Estocástico (Oscilador Estocástico)
        # O estocástico compara o preço de fechamento com o intervalo de preços em um determinado período de tempo.
        stoch_k, stoch_d = ta.STOCH(df['high'], df['low'], df['close'],
                                    fastk_period=14, slowk_period=3, slowk_matype=0,
                                    slowd_period=3, slowd_matype=0)
        df['stoch_k'] = stoch_k  # Linha %K
        df['stoch_d'] = stoch_d  # Linha %D

        # 14. ADX (Average Directional Index)
        # O ADX mede a força de uma tendência. Valores mais altos indicam tendências mais fortes.
        df['adx_14'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # 15. CCI (Commodity Channel Index)
        # O CCI mede a variação do preço de um ativo em relação à sua média histórica.
        df['cci_14'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=14)

        # 16. Chande Momentum Oscillator
        # O CMO é um indicador de momentum que mede a velocidade da mudança de preços.
        df['cmo_14'] = ta.CMO(df['close'], timeperiod=14)

        # 17. Vortex Indicator (usando função manual)
        # O Vortex é um indicador que mede a direção e a intensidade da tendência.
        df = vortex_indicator(df, period=14)

        # 18. Elder Ray Index
        # O Elder Ray calcula a diferença entre os preços máximos e a média exponencial de 13 períodos (EMA).
        # O índice identifica a força dos touros (bulls) e ursos (bears).
        df['ema_13'] = ta.EMA(df['close'], 13)
        df['bull_power'] = df['high'] - df['ema_13']  # Poder dos touros
        df['bear_power'] = df['low'] - df['ema_13']  # Poder dos ursos

        # 19. Divergência MACD usando Z-Score
        # A divergência do MACD usando o Z-Score é uma métrica que pode ser utilizada para identificar sinais de reversão
        # com base nas mudanças no histograma do MACD em relação ao preço.
        # (Este cálculo está comentado, mas pode ser descomentado e usado)
        # df['macd_divergence'] = zscore(df['macd_hist']) - zscore(df['close'])

    except Exception as e:
        # Caso ocorra um erro, ele será registrado no log de exceções
        logging.exception("Erro em add_momentum_indicators: %s", e)

    # Retorna o DataFrame com as novas colunas de indicadores de momentum
    return df


def add_volume_indicators(df):
    """
    Adiciona indicadores baseados em volume ao DataFrame, como o Chaikin Money Flow (CMF),
    Money Flow Index (MFI), VWAP, OBV, Volume Rate of Change (VROC), Volume Profile e Accumulation/Distribution Line (ADL).

    Esses indicadores são úteis para analisar o fluxo de volume em relação aos preços e ajudar a identificar
    movimentos significativos de compra ou venda no mercado.

    Parâmetros:
        df: DataFrame contendo os dados de preço (close, high, low) e volume (tick_volume ou volume).

    Retorna:
        df: DataFrame com os novos indicadores baseados em volume adicionados.
    """
    try:
        # 20. Definindo a coluna de volume: 'tick_volume' se estiver presente, caso contrário, usa 'volume'
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'

        # 21. Chaikin Money Flow (CMF)
        # O CMF calcula o fluxo de dinheiro de um ativo, considerando o volume e a relação entre os preços de fechamento, máximos e mínimos.
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0)  # Substitui valores infinitos e NaN
        mfv = mfm * df[vol_col]  # Multiplica o Money Flow Multiplier pelo volume
        # Soma o fluxo de dinheiro (MFV) nos últimos 20 períodos e calcula a média ponderada
        df['cmf_20'] = mfv.rolling(window=20).sum() / df[vol_col].rolling(window=20).sum()

        # 22. Money Flow Index (MFI)
        # O MFI mede a pressão de compra e venda considerando o preço e o volume, ajudando a identificar sobrecompra ou sobrevenda.
        df['mfi_10'] = ta.MFI(df['high'], df['low'], df['close'], df[vol_col], timeperiod=10)

        # 23. VWAP (Volume Weighted Average Price)
        # O VWAP calcula o preço médio ponderado pelo volume e é comumente usado em negociações intraday.
        df['vwap_20'] = (df['close'] * df[vol_col]).rolling(window=20).sum() / df[vol_col].rolling(window=20).sum()

        # 24. On-Balance Volume (OBV)
        # O OBV é um indicador que acumula o volume com base no movimento do preço: positivo se o preço subir, negativo se o preço cair.
        df['obv'] = ta.OBV(df['close'], df[vol_col])

        # 25. Volume Rate of Change (VROC)
        # O VROC mede a variação percentual do volume em relação ao volume de períodos anteriores, indicando mudanças no fluxo de volume.
        df['vroc_14'] = df[vol_col].pct_change(periods=14) * 100

        # 26. Volume Profile
        # O Volume Profile é uma medida do volume negociado em determinado intervalo de preços, que pode indicar áreas de suporte e resistência.
        df['volume_profile'] = df[vol_col] * (df['high'] - df['low'])

        # 27. Accumulation/Distribution Line (ADL)
        # O ADL indica o fluxo de dinheiro acumulado ao longo do tempo, com base no volume e na variação de preço.
        df['adl'] = ta.ADOSC(df['high'], df['low'], df['close'], df[vol_col], fastperiod=3, slowperiod=10)

        # 28. VWAP RSI
        # O VWAP RSI combina o RSI com o VWAP, dando mais peso aos preços ponderados pelo volume para calcular o RSI.
        df['vw_rsi'] = (df['rsi_14'] * df[vol_col]) / df[vol_col].rolling(14).mean()

    except Exception as e:
        # Caso ocorra um erro, ele será registrado no log de exceções
        logging.exception("Erro em add_volume_indicators: %s", e)

    # Retorna o DataFrame com as novas colunas de indicadores baseados em volume
    return df


def add_crossover_indicators(df):
    """
    Adiciona indicadores de cruzamento de médias móveis simples (SMA) ao DataFrame.

    Os cruzamentos de médias móveis são usados para identificar mudanças de tendência e gerar sinais de compra ou venda.
    O cruzamento da SMA de períodos mais curtos com as de períodos mais longos pode indicar **mudanças significativas** na direção do preço.

    Parâmetros:
        df: DataFrame contendo os dados de médias móveis simples (SMA).

    Retorna:
        df: DataFrame com as novas colunas de indicadores de cruzamento de médias móveis adicionadas.
    """
    try:
        # 29. Cruzamento de SMA 50 e 200
        # Calcula a diferença entre as médias móveis de 50 e 200 períodos
        df['sma_diff_50_200'] = df['sma_50'] - df['sma_200']

        # A coluna 'sma_cross_signal_50_200' indica o cruzamento entre as SMAs de 50 e 200 períodos.
        # Se a diferença (sma_diff_50_200) for positiva, isso significa que a SMA de 50 está acima da SMA de 200,
        # indicando uma tendência de alta e um sinal de compra (1).
        # Se for negativa, a SMA de 50 está abaixo da SMA de 200, indicando uma tendência de baixa e um sinal de venda (-1).
        # A função `diff()` calcula a diferença entre o valor atual e o anterior para identificar as mudanças de direção.
        df['sma_cross_signal_50_200'] = df['sma_diff_50_200'].apply(lambda x: 1 if x > 0 else -1).diff()

        # 30. Cruzamento de SMA 9 e 21
        # Calcula a diferença entre as médias móveis de 9 e 21 períodos
        df['sma_diff_9_21'] = df['sma_9'] - df['sma_21']

        # A coluna 'sma_cross_signal_9_21' indica o cruzamento entre as SMAs de 9 e 21 períodos.
        # Se a diferença (sma_diff_9_21) for positiva, isso significa que a SMA de 9 está acima da SMA de 21,
        # indicando uma tendência de alta e um sinal de compra (1).
        # Se for negativa, a SMA de 9 está abaixo da SMA de 21, indicando uma tendência de baixa e um sinal de venda (-1).
        # A função `diff()` é utilizada para detectar o momento do cruzamento entre as SMAs.
        df['sma_cross_signal_9_21'] = df['sma_diff_9_21'].apply(lambda x: 1 if x > 0 else -1).diff()

    except Exception as e:
        # Caso ocorra um erro, ele será registrado no log de exceções
        logging.exception("Erro em add_crossover_indicators: %s", e)

    # Retorna o DataFrame com as novas colunas de cruzamento de médias móveis
    return df



def add_lag_features(df, lags=[1, 2, 3]):
    """
    Adiciona as colunas de lags para os preços low, high, open e close de uma vez.

    Os **lags** são usados para incluir as observações anteriores de uma série temporal
    como novas variáveis, o que é útil para **modelos de previsão** e **análises de dependência temporal**.

    Parâmetros:
        df: DataFrame contendo as colunas de preço (low, high, open, close).
        lags: Lista de períodos (lags) para os quais as colunas de preços serão deslocadas.

    Retorna:
        df: DataFrame com as novas colunas de lags adicionadas.
    """
    try:
        lag_columns = []

        # Adicionando os lags das colunas de preços (low, high, open, close)
        # A função `shift(i)` é usada para deslocar os dados para i períodos atrás.
        for i in range(1, 4):  # Para os períodos de 1 a 3 (últimos 3 períodos)
            lag_columns.append(df['low'].shift(i).rename(f'low_lag{i}'))  # Deslocando os preços de "low"
            lag_columns.append(df['high'].shift(i).rename(f'high_lag{i}'))  # Deslocando os preços de "high"
            lag_columns.append(df['open'].shift(i).rename(f'open_lag{i}'))  # Deslocando os preços de "open"
            lag_columns.append(df['close'].shift(i).rename(f'close_lag{i}'))  # Deslocando os preços de "close"

        # Concatenando todas as colunas de lag ao DataFrame original
        # `pd.concat([df] + lag_columns, axis=1)` combina o DataFrame original com as novas colunas de lags.
        df = pd.concat([df] + lag_columns, axis=1)

    except Exception as e:
        # Caso ocorra um erro, ele será registrado no log de exceções
        logging.exception("Erro em add_lag_features: %s", e)

    # Retorna o DataFrame com as novas colunas de lags
    return df


def add_hull_moving_average(df):
    """
    Adiciona as médias móveis de Hull (HMA) ao DataFrame para diferentes períodos (9, 14, 21, 50 e 100).

    A média móvel de Hull (HMA) é uma versão suavizada da média móvel ponderada, que visa reduzir o
    atraso e melhorar a resposta às mudanças de preço. A HMA combina médias móveis ponderadas (WMA)
    para calcular uma média móvel mais eficiente, permitindo uma análise de tendência mais precisa.

    Parâmetros:
        df: DataFrame contendo os dados de preços (close).

    Retorna:
        df: DataFrame com as novas colunas de médias móveis de Hull (HMA) adicionadas.
    """
    try:
        # Calculando as médias móveis de Hull para diferentes períodos (9, 14, 21, 50 e 100)
        hma_9 = hull_ma(df['close'], 9)  # HMA de 9 períodos
        hma_14 = hull_ma(df['close'], 14)  # HMA de 14 períodos
        hma_21 = hull_ma(df['close'], 21)  # HMA de 21 períodos
        hma_50 = hull_ma(df['close'], 50)  # HMA de 50 períodos
        hma_100 = hull_ma(df['close'], 100)  # HMA de 100 períodos

        # Concatenando as novas colunas de HMA de uma vez para evitar múltiplas inserções no DataFrame
        # O método `pd.concat` adiciona as novas colunas ao DataFrame original.
        df = pd.concat([df, pd.DataFrame({'hma_9': hma_9, 'hma_14': hma_14, 'hma_21': hma_21, 'hma_50': hma_50, 'hma_100': hma_100})], axis=1)

    except Exception as e:
        # Caso ocorra um erro durante o cálculo das HMAs, o erro será registrado no log de exceções
        logging.exception("Erro em add_hull_moving_average: %s", e)

    # Retorna o DataFrame com as novas colunas de HMA adicionadas
    return df


def add_risk_features(df):
    """
    Adiciona indicadores de risco ao DataFrame, incluindo o **Drawdown**, **Value at Risk (VaR)** de 95% e **Estatísticas de ADF**.

    Esses indicadores são utilizados para avaliar o **risco de mercado** e a **exposição a perdas**,
    ajudando a entender o comportamento do ativo em momentos de **queda acentuada** ou **volatilidade**.

    Parâmetros:
        df: DataFrame contendo os dados de preço (close) e retornos logarítmicos (log_ret).

    Retorna:
        df: DataFrame com os novos indicadores de risco calculados.
    """
    try:
        # Calculando o **cumulative max** (máximo acumulado) para os preços de fechamento
        cum_max = df['close'].cummax()  # Valor máximo acumulado dos preços de fechamento

        # Calculando o **drawdown** (redução máxima) em relação ao valor máximo acumulado
        # O drawdown é a diferença entre o valor máximo acumulado e o preço de fechamento atual,
        # normalizado pelo valor máximo acumulado.
        drawdown = (cum_max - df['close']) / cum_max  # Proporção de perda em relação ao valor máximo

        # Calculando o **VaR 95%** (Value at Risk) usando o **percentil 5** dos retornos logarítmicos
        # O VaR é uma medida de risco que calcula a **perda potencial** com uma confiança de 95%.
        var_95 = df['log_ret'].rolling(20).apply(lambda x: np.percentile(x, 5), raw=True)

        # Concatenando os novos indicadores de risco em um DataFrame
        df_risk = pd.DataFrame({
            'cum_max': cum_max,  # Máximo acumulado
            'drawdown': drawdown,  # Drawdown
            'var_95': var_95,  # VaR 95%
            # Estatística de ADF (Augmented Dickey-Fuller) com janela de 30 períodos
            # O teste ADF é usado para verificar a **estacionaridade** da série temporal (se a série tem ou não uma raiz unitária)
            'adf_stat_30': df['close'].rolling(30).apply(rolling_adf, raw=True)
        })

        # Concatenando as novas colunas de risco ao DataFrame original
        # `pd.concat` adiciona as novas colunas de risco ao DataFrame original.
        df = pd.concat([df, df_risk], axis=1)

    except Exception as e:
        # Caso ocorra um erro durante o cálculo dos indicadores de risco, o erro será registrado no log de exceções
        logging.exception("Erro em add_risk_features: %s", e)

    # Retorna o DataFrame com os novos indicadores de risco adicionados
    return df


def vortex_indicator(df, period=14):
    """
    Calcula o indicador Vortex (+VI e -VI) manualmente.

    O indicador Vortex é usado para medir a **direção e a intensidade da tendência** de um ativo.
    Ele consiste em duas linhas: **+VI** e **-VI**, que representam a força da tendência de alta e baixa, respectivamente.

    Parâmetros:
        df: DataFrame contendo os dados de preço (high, low, close).
        period: O período utilizado para calcular o Vortex (padrão é 14).

    Retorna:
        df: DataFrame com as novas colunas de indicadores Vortex (+VI e -VI) adicionadas.
    """

    # Cálculo do **True Range (TR)**: O True Range é a maior diferença entre:
    # 1. A diferença entre o preço máximo e o preço mínimo do período.
    # 2. A diferença absoluta entre o preço máximo do período atual e o preço de fechamento do período anterior.
    # 3. A diferença absoluta entre o preço mínimo do período atual e o preço de fechamento do período anterior.
    df['TR'] = pd.concat([df['high'] - df['low'],
                          (df['high'] - df['close'].shift()).abs(),
                          (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)

    # Cálculo do **Positive Directional Movement (+DM)**:
    # O **+DM** é a diferença entre o preço máximo do período atual e o preço máximo do período anterior.
    # Se o valor for positivo, significa que a tendência é de alta; caso contrário, o valor é zero.
    df['+DM'] = df['high'] - df['high'].shift()
    df['+DM'] = df['+DM'].apply(lambda x: x if x > 0 else 0)

    # Cálculo do **Negative Directional Movement (-DM)**:
    # O **-DM** é a diferença entre o preço mínimo do período anterior e o preço mínimo do período atual.
    # Se o valor for positivo, significa que a tendência é de baixa; caso contrário, o valor é zero.
    df['-DM'] = df['low'].shift() - df['low']
    df['-DM'] = df['-DM'].apply(lambda x: x if x > 0 else 0)

    # Cálculo das somas acumuladas para **TR**, **+DM** e **-DM**:
    # As somas acumuladas são feitas para um **período (window)** de 14 períodos, usando a função `rolling().sum()`.
    df['+DM_sum'] = df['+DM'].rolling(window=period).sum()  # Soma acumulada de +DM
    df['-DM_sum'] = df['-DM'].rolling(window=period).sum()  # Soma acumulada de -DM
    df['TR_sum'] = df['TR'].rolling(window=period).sum()    # Soma acumulada de TR

    # Cálculo do **Vortex (+VI e -VI)**:
    # O **+VI** é calculado dividindo a soma acumulada do **+DM** pela soma acumulada do **TR**, multiplicado por 100.
    # O **-VI** é calculado dividindo a soma acumulada do **-DM** pela soma acumulada do **TR**, multiplicado por 100.
    df['+VI'] = df['+DM_sum'] / df['TR_sum'] * 100  # +VI (força da tendência de alta)
    df['-VI'] = df['-DM_sum'] / df['TR_sum'] * 100  # -VI (força da tendência de baixa)

    # Retorna o DataFrame com as novas colunas de +VI e -VI
    return df


def add_additional_metrics(df, market_log_ret=None):
    """
    Adiciona métricas financeiras adicionais ao DataFrame, como Sharpe Ratio, Sortino Ratio, Beta,
    correlação com o mercado, entre outras métricas para análise de risco e desempenho financeiro.

    Essas métricas são comumente usadas para avaliar o risco e o retorno de um ativo financeiro.

    Parâmetros:
        df: DataFrame contendo os dados de preços (close) e retornos logarítmicos (log_ret).
        market_log_ret: Retornos logarítmicos do mercado (opcional, utilizado para calcular Beta e correlação).

    Retorna:
        df: DataFrame com as novas métricas financeiras adicionadas.
    """
    try:
        # 1. Pré-processamento dos dados
        # ----------------------------------------------------------------------
        # Criar uma cópia para evitar modificações no DataFrame original
        df = df.copy()

        # Garantir que 'log_ret' (retorno logarítmico) exista no DataFrame
        if 'log_ret' not in df.columns:
            # Calcular os retornos logarítmicos como log(close / close.shift(1))
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        # 2. Cálculo de métricas base
        # ----------------------------------------------------------------------
        # **Max Drawdown**: A maior perda em relação ao valor máximo histórico
        df['max_drawdown'] = df['close'].cummax() - df['close']  # Diferença entre o máximo acumulado e o preço atual

        # **Volatilidade de 30 dias**: O desvio padrão dos retornos logarítmicos dos últimos 30 dias
        df['volatility_30'] = df['log_ret'].rolling(window=30).std()

        # 3. Cálculo de **ratios** financeiros (com tratamento de divisão por zero)
        # ----------------------------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            # **Sharpe Ratio**: Mede o retorno ajustado pelo risco (quantidade de retorno por unidade de risco)
            sharpe = np.where(df['volatility_30'] != 0,
                            df['log_ret'].rolling(window=30).mean() / df['volatility_30'],
                            np.nan)  # Se a volatilidade for 0, o Sharpe é indefinido

            # **Sortino Ratio**: Uma variação do Sharpe que considera apenas a volatilidade negativa (perdas)
            negative_returns = df['log_ret'].where(df['log_ret'] < 0)  # Filtra retornos negativos
            sortino_denominator = negative_returns.rolling(window=30).std()  # Desvio padrão dos retornos negativos
            sortino = np.where(sortino_denominator != 0,
                             df['log_ret'].rolling(window=30).mean() / sortino_denominator,
                             np.nan)

        # 4. Cálculo de **Beta** e **Correlação** com o mercado
        # ----------------------------------------------------------------------
        # Se dados de mercado forem fornecidos, calcula o Beta (risco sistemático) e a correlação com o mercado
        beta = pd.Series(index=df.index, dtype=float)  # Cria uma série vazia para Beta
        correlation = pd.Series(index=df.index, dtype=float)  # Cria uma série vazia para correlação

        if market_log_ret is not None:
            # Alinha as séries temporais do ativo e do mercado para garantir que tenham os mesmos índices
            aligned_data = pd.DataFrame({
                'asset': df['log_ret'],
                'market': market_log_ret
            }).dropna()  # Remove valores ausentes

            # Calcula a **covariância** entre o ativo e o mercado para uma janela de 30 períodos
            covariance = aligned_data['asset'].rolling(window=30).cov(aligned_data['market'])
            # Calcula a **variância** do mercado para uma janela de 30 períodos
            market_variance = aligned_data['market'].rolling(window=30).var()

            # **Beta**: A covariância do ativo com o mercado dividida pela variância do mercado
            beta_update = covariance / market_variance
            beta.update(beta_update)

            # **Correlação**: A correlação do ativo com o mercado para uma janela de 30 períodos
            correlation_update = aligned_data['asset'].rolling(window=30).corr(aligned_data['market'])
            correlation.update(correlation_update)

        # 5. Adicionar as métricas ao DataFrame
        # ----------------------------------------------------------------------
        df['sharpe_ratio'] = sharpe  # Adiciona o Sharpe Ratio
        df['sortino_ratio'] = sortino  # Adiciona o Sortino Ratio
        df['beta'] = beta  # Adiciona o Beta
        df['market_correlation'] = correlation  # Adiciona a correlação com o mercado

        # 6. Cálculo de métricas adicionais
        # ----------------------------------------------------------------------
        # **EWMA (Exponential Weighted Moving Average)** de 30 períodos
        df['ewma_30'] = df['close'].ewm(span=30, adjust=False).mean()

        # **Assimetria (Skewness)**: Mede a assimetria dos retornos logarítmicos, se a distribuição é assimétrica
        df['returns_skewness'] = df['log_ret'].rolling(window=30).skew()

        # **Curtose (Kurtosis)**: Mede a "cauda" da distribuição dos retornos, se tem mais ou menos outliers
        df['returns_kurtosis'] = df['log_ret'].rolling(window=30).kurt()

        # **Ulcer Index**: Mede a severidade do drawdown ao longo do tempo, ajudando a avaliar o risco de grandes quedas
        squared_drawdown = df['max_drawdown']**2
        df['ulcer_index'] = np.sqrt(squared_drawdown.rolling(window=252).mean())  # Média móvel do drawdown ao longo de 252 dias

    except Exception as e:
        # Caso ocorra um erro durante o cálculo, o erro será registrado no log de exceções
        logging.exception("Erro em add_additional_metrics: %s", e)

    # Retorna o DataFrame com as novas métricas financeiras adicionais
    return df

def calculate_moving_average(df, column_name, periods, ma_type="SMA"):
    """
    Calcula médias móveis (SMA ou EMA) para a coluna especificada.

    Parâmetros:
        df: DataFrame contendo os dados.
        column_name: Nome da coluna para a qual calcular a média móvel.
        periods: Lista ou valor único representando o período da média móvel.
        ma_type: Tipo de média móvel a ser calculado, pode ser 'SMA' (média móvel simples) ou 'EMA' (média móvel exponencial).

    Retorna:
        df: DataFrame com as novas colunas de médias móveis calculadas.
    """
    if isinstance(periods, list):
        for period in periods:
            if ma_type == "SMA":
                df[f"sma_{period}"] = ta.SMA(df[column_name], timeperiod=period)
            elif ma_type == "EMA":
                df[f"ema_{period}"] = ta.EMA(df[column_name], timeperiod=period)
            else:
                raise ValueError("Tipo de média móvel inválido. Use 'SMA' ou 'EMA'.")
    else:
        period = periods
        if ma_type == "SMA":
            df[f"sma_{period}"] = ta.SMA(df[column_name], timeperiod=period)
        elif ma_type == "EMA":
            df[f"ema_{period}"] = ta.EMA(df[column_name], timeperiod=period)
        else:
            raise ValueError("Tipo de média móvel inválido. Use 'SMA' ou 'EMA'.")

    return df
