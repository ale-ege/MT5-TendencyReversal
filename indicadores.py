import talib as ta
import pandas as pd
import numpy as np
import logging
from scipy.stats import zscore
from statsmodels.tsa.stattools import adfuller

def calcular_indicadores(df, symbol):
    try:
        # Adiciona novas informações
        df['bid'] = [mt5.symbol_info_tick(symbol).bid for _ in range(len(df))]
        df['ask'] = [mt5.symbol_info_tick(symbol).ask for _ in range(len(df))]
        df['last_price'] = (df['bid'] + df['ask']) / 2
        df['volatility'] = df['high'] - df['low']
        df['change_percent'] = ((df['close'] - df['open']) / df['open']) * 100
        df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # Indicadores técnicos adicionais
        df['moving_avg_9'] = ta.SMA(df['close'], timeperiod=9)
        df['moving_avg_21'] = ta.SMA(df['close'], timeperiod=21)
        df['moving_avg_50'] = ta.SMA(df['close'], timeperiod=50)
        df['ema_12'] = ta.EMA(df['close'], timeperiod=12)
        df['ema_26'] = ta.EMA(df['close'], timeperiod=26)
        df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)

        macd, signal_macd, _ = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['signal_macd'] = signal_macd

        upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20)
        df['bb_upper'] = upper
        df['bb_lower'] = lower

        # Novos Indicadores Técnicos
        df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)  # Average True Range
        df['roc_10'] = ta.ROC(df['close'], timeperiod=10)  # Rate of Change
        df['obv'] = ta.OBV(df['close'], df['tick_volume'])  # On-Balance Volume
        df['mfi_14'] = ta.MFI(df['high'], df['low'], df['close'], df['tick_volume'], timeperiod=14)  # Money Flow Index

        # Estatísticas de volatilidade e momentum
        df['rolling_max_20'] = df['close'].rolling(window=20).max()
        df['rolling_min_20'] = df['close'].rolling(window=20).min()
        df['std_dev_20'] = df['close'].rolling(window=20).std()

        # **ADX (Average Directional Index) - Força da Tendência**
        df['adx_14'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # **VWAP (Volume Weighted Average Price)**
        df['vwap'] = (df['close'] * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()

        # **Média do OBV em diferentes períodos**
        df['obv_10'] = df['obv'].rolling(window=10).mean()
        df['obv_20'] = df['obv'].rolling(window=20).mean()
        df['obv_50'] = df['obv'].rolling(window=50).mean()

    except Exception as e:
        logging.exception("Erro em calcular_indicadores: %s", e)

    return df


def add_basic_features(df):
    try:
        # 1. Volatilidade
        df['volatility'] = df['high'] - df['low']

        # 2. Percentual de mudança no preço
        df['change_percent'] = ((df['close'] - df['open']) / df['open']) * 100

        # 3. Preço médio (arredondado)
        df['avg_price'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4

        # 4. Retorno logarítmico
        df['log_ret'] = np.log(df['close'] / df['close'].shift(1))  # Retornos logarítmicos

        # 5. Z-score de volatilidade
        df['zscore_vol'] = df['volatility'].rolling(window=50).apply(
            lambda x: (x[-1] - x.mean()) / x.std(), raw=True)

    except Exception as e:
        logging.exception("Erro em add_basic_features: %s", e)
    return df


def hull_ma(series, window):
    """Calcula a Hull Moving Average (HMA)."""
    wma_half = ta.WMA(series, window // 2)  # Média móvel ponderada de metade do período
    wma_full = ta.WMA(series, window)  # Média móvel ponderada do período completo
    hma_series = ta.WMA(2 * wma_half - wma_full, int(np.sqrt(window)))  # Resultado da HMA
    return hma_series


def rolling_adf(x):
    """Aplica o teste ADF e retorna a estatística de teste."""
    result = adfuller(x)
    return result[0]  # Retorna a estatística de teste ADF


def add_moving_averages(df):
    try:
        # 6. Médias móveis
        df['sma_9'] = ta.SMA(df['close'], timeperiod=9)
        df['sma_21'] = ta.SMA(df['close'], timeperiod=21)
        df['sma_50'] = ta.SMA(df['close'], timeperiod=50)
        df['sma_100'] = ta.SMA(df['close'], timeperiod=100)
        df['sma_200'] = ta.SMA(df['close'], timeperiod=200)

    except Exception as e:
        logging.exception("Erro em add_moving_averages: %s", e)
    return df


def add_volatility_indicators(df):
    try:
        # 7. ATR de 14 e 50 períodos
        df['atr_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
        df['atr_50'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=50)

        # 8. Volatilidade Rolling de 20 períodos
        df['rolling_volatility_20'] = df['close'].rolling(window=20).std()

        # 9. Bandas de Bollinger
        upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
        df['bb_width'] = upper - lower
        df['bollinger_percent_b'] = (df['close'] - lower) / (upper - lower)

        # 10. Keltner Channels
        df['ema_20'] = ta.EMA(df['close'], timeperiod=20)
        df['keltner_upper'] = df['ema_20'] + 2 * df['atr_14']
        df['keltner_lower'] = df['ema_20'] - 2 * df['atr_14']
        df['keltner_break'] = np.where(df['close'] > df['keltner_upper'], 1,
                                       np.where(df['close'] < df['keltner_lower'], -1, 0))

    except Exception as e:
        logging.exception("Erro em add_volatility_indicators: %s", e)
    return df


def add_momentum_indicators(df):
    try:
        # 11. Indicadores de momentum
        df['roc_5'] = ta.ROC(df['close'], timeperiod=5)
        df['roc_20'] = ta.ROC(df['close'], timeperiod=20)
        df['williams_r_14'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
        df['momentum_10'] = ta.MOM(df['close'], timeperiod=10)
        df['rsi_14'] = ta.RSI(df['close'], timeperiod=14)

        # 12. MACD
        macd, macd_signal, macd_hist = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
        df['macd'] = macd
        df['macd_signal'] = macd_signal
        df['macd_hist'] = macd_hist

        # 13. Estocástico
        stoch_k, stoch_d = ta.STOCH(df['high'], df['low'], df['close'],
                                    fastk_period=14, slowk_period=3, slowk_matype=0,
                                    slowd_period=3, slowd_matype=0)
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d

        # 14. ADX
        df['adx_14'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        # 15. CCI
        df['cci_14'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=14)

        # 16. Chande Momentum Oscillator
        df['cmo_14'] = ta.CMO(df['close'], timeperiod=14)

        # 17. Vortex (usando função manual)
        df = vortex_indicator(df, period=14)

        # 18. Elder Ray Index
        df['ema_13'] = ta.EMA(df['close'], 13)
        df['bull_power'] = df['high'] - df['ema_13']
        df['bear_power'] = df['low'] - df['ema_13']

        # 19. Divergência MACD usando Z-Score
        #df['macd_divergence'] = zscore(df['macd_hist']) - zscore(df['close'])

    except Exception as e:
        logging.exception("Erro em add_momentum_indicators: %s", e)
    return df

def add_volume_indicators(df):
    try:
        # 20. Indicadores baseados em volume
        vol_col = 'tick_volume' if 'tick_volume' in df.columns else 'volume'

        # 21. Chaikin Money Flow (CMF)
        mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfm = mfm.replace([np.inf, -np.inf], np.nan).fillna(0)
        mfv = mfm * df[vol_col]
        df['cmf_20'] = mfv.rolling(window=20).sum() / df[vol_col].rolling(window=20).sum()

        # 22. Money Flow Index (MFI)
        df['mfi_10'] = ta.MFI(df['high'], df['low'], df['close'], df[vol_col], timeperiod=10)

        # 23. VWAP
        df['vwap_20'] = (df['close'] * df[vol_col]).rolling(window=20).sum() / df[vol_col].rolling(window=20).sum()

        # 24. On-Balance Volume (OBV)
        df['obv'] = ta.OBV(df['close'], df[vol_col])

        # 25. Volume Rate of Change (VROC)
        df['vroc_14'] = df[vol_col].pct_change(periods=14) * 100

        # 26. Volume Profile
        df['volume_profile'] = df[vol_col] * (df['high'] - df['low'])

        # 27. Accumulation/Distribution Line (ADL)
        df['adl'] = ta.ADOSC(df['high'], df['low'], df['close'], df[vol_col], fastperiod=3, slowperiod=10)

        # 28. VWAP RSI
        df['vw_rsi'] = (df['rsi_14'] * df[vol_col]) / df[vol_col].rolling(14).mean()

    except Exception as e:
        logging.exception("Erro em add_volume_indicators: %s", e)
    return df


def add_crossover_indicators(df):
    try:
        # 29. Cruzamento de SMA 50 e 200
        df['sma_diff_50_200'] = df['sma_50'] - df['sma_200']
        df['sma_cross_signal_50_200'] = df['sma_diff_50_200'].apply(lambda x: 1 if x > 0 else -1).diff()

        # 30. Cruzamento de SMA 9 e 21
        df['sma_diff_9_21'] = df['sma_9'] - df['sma_21']
        df['sma_cross_signal_9_21'] = df['sma_diff_9_21'].apply(lambda x: 1 if x > 0 else -1).diff()

    except Exception as e:
        logging.exception("Erro em add_crossover_indicators: %s", e)
    return df


def add_lag_features(df, lags=[1, 2, 3]):
    """Adiciona as colunas de lags para os preços low, high, open e close de uma vez."""
    try:
        lag_columns = []

        # Adicionando os lags das colunas de preços
        for i in range(1, 4):  # Para os últimos 21 períodos
            lag_columns.append(df['low'].shift(i).rename(f'low_lag{i}'))
            lag_columns.append(df['high'].shift(i).rename(f'high_lag{i}'))
            lag_columns.append(df['open'].shift(i).rename(f'open_lag{i}'))
            lag_columns.append(df['close'].shift(i).rename(f'close_lag{i}'))

        # Concatenando todas as colunas de lag de uma vez
        df = pd.concat([df] + lag_columns, axis=1)

    except Exception as e:
        logging.exception("Erro em add_lag_features: %s", e)
    return df
def add_hull_moving_average(df):
    try:
        # Calculando as médias móveis de Hull
        hma_14 = hull_ma(df['close'], 14)
        hma_50 = hull_ma(df['close'], 50)

        # Concatenando as novas colunas de uma vez para evitar múltiplas inserções
        df = pd.concat([df, pd.DataFrame({'hma_14': hma_14, 'hma_50': hma_50})], axis=1)

    except Exception as e:
        logging.exception("Erro em add_hull_moving_average: %s", e)
    return df

def add_risk_features(df):
    try:
        # Calculando os indicadores de risco
        cum_max = df['close'].cummax()
        drawdown = (cum_max - df['close']) / cum_max
        var_95 = df['log_ret'].rolling(20).apply(lambda x: np.percentile(x, 5), raw=True)

        # Concatenando as novas colunas de uma vez para evitar múltiplas inserções
        df_risk = pd.DataFrame({
            'cum_max': cum_max,
            'drawdown': drawdown,
            'var_95': var_95,
            'adf_stat_30': df['close'].rolling(30).apply(rolling_adf, raw=True)
        })

        # Concatenando as novas colunas ao DataFrame original
        df = pd.concat([df, df_risk], axis=1)

    except Exception as e:
        logging.exception("Erro em add_risk_features: %s", e)
    return df



def vortex_indicator(df, period=14):
    """Calcula o indicador Vortex (+VI e -VI) manualmente."""

    # Cálculo do True Range (TR)
    df['TR'] = pd.concat([df['high'] - df['low'],
                          (df['high'] - df['close'].shift()).abs(),
                          (df['low'] - df['close'].shift()).abs()], axis=1).max(axis=1)

    # Cálculo do Positive Directional Movement (+DM)
    df['+DM'] = df['high'] - df['high'].shift()
    df['+DM'] = df['+DM'].apply(lambda x: x if x > 0 else 0)

    # Cálculo do Negative Directional Movement (-DM)
    df['-DM'] = df['low'].shift() - df['low']
    df['-DM'] = df['-DM'].apply(lambda x: x if x > 0 else 0)

    # Cálculo das somas acumuladas para TR, +DM, -DM
    df['+DM_sum'] = df['+DM'].rolling(window=period).sum()
    df['-DM_sum'] = df['-DM'].rolling(window=period).sum()
    df['TR_sum'] = df['TR'].rolling(window=period).sum()

    # Cálculo do Vortex (+VI e -VI)
    df['+VI'] = df['+DM_sum'] / df['TR_sum'] * 100
    df['-VI'] = df['-DM_sum'] / df['TR_sum'] * 100

    return df


def add_additional_metrics(df, market_log_ret=None):
    """Adiciona métricas financeiras adicionais ao DataFrame."""
    try:
        # 1. Pré-processamento dos dados
        # ----------------------------------------------------------------------
        # Criar cópia para evitar modificações no DataFrame original
        df = df.copy()

        # Garantir que 'log_ret' existe
        if 'log_ret' not in df.columns:
            df['log_ret'] = np.log(df['close'] / df['close'].shift(1))

        # 2. Cálculo de métricas base
        # ----------------------------------------------------------------------
        # Drawdown e volatilidade
        df['max_drawdown'] = df['close'].cummax() - df['close']
        df['volatility_30'] = df['log_ret'].rolling(window=30).std()

        # 3. Cálculo de ratios (com tratamento de divisão por zero)
        # ----------------------------------------------------------------------
        with np.errstate(divide='ignore', invalid='ignore'):
            # Sharpe Ratio
            sharpe = np.where(df['volatility_30'] != 0,
                            df['log_ret'].rolling(window=30).mean() / df['volatility_30'],
                            np.nan)

            # Sortino Ratio
            negative_returns = df['log_ret'].where(df['log_ret'] < 0)
            sortino_denominator = negative_returns.rolling(window=30).std()
            sortino = np.where(sortino_denominator != 0,
                             df['log_ret'].rolling(window=30).mean() / sortino_denominator,
                             np.nan)

        # 4. Cálculo de Beta e Correlação (se houver dados de mercado)
        # ----------------------------------------------------------------------
        beta = pd.Series(index=df.index, dtype=float)
        correlation = pd.Series(index=df.index, dtype=float)

        if market_log_ret is not None:
            # Alinhar séries temporalmente
            aligned_data = pd.DataFrame({
                'asset': df['log_ret'],
                'market': market_log_ret
            }).dropna()

            # Calcular covariância e variância de forma vetorizada
            covariance = aligned_data['asset'].rolling(window=30).cov(aligned_data['market'])
            market_variance = aligned_data['market'].rolling(window=30).var()

            beta_update = covariance / market_variance
            beta.update(beta_update)

            correlation_update = aligned_data['asset'].rolling(window=30).corr(aligned_data['market'])
            correlation.update(correlation_update)

        # 5. Adicionar métricas ao DataFrame
        # ----------------------------------------------------------------------
        df['sharpe_ratio'] = sharpe
        df['sortino_ratio'] = sortino
        df['beta'] = beta
        df['market_correlation'] = correlation

        # 6. Métricas adicionais
        # ----------------------------------------------------------------------
        # EWMA e outras métricas
        df['ewma_30'] = df['close'].ewm(span=30, adjust=False).mean()
        df['returns_skewness'] = df['log_ret'].rolling(window=30).skew()
        df['returns_kurtosis'] = df['log_ret'].rolling(window=30).kurt()

        # Ulcer Index
        squared_drawdown = df['max_drawdown']**2
        df['ulcer_index'] = np.sqrt(squared_drawdown.rolling(window=252).mean())

    except Exception as e:
        logging.exception("Erro em add_additional_metrics: %s", e)

    return df
