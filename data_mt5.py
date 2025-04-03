import MetaTrader5 as mt5  # Importa a biblioteca MetaTrader5 para interagir com o MetaTrader 5 (MT5) e obter dados do mercado.
import pandas as pd  # Importa o Pandas para manipulação de dados, como DataFrames.
import logging  # Importa o módulo de logging para registrar informações e erros.
import numpy as np  # Importa o NumPy para cálculos numéricos, como operações com arrays.
import os  # Importa o módulo os para interagir com o sistema de arquivos (criação de diretórios, etc.)

# Importando a variável de configuração de config.py
from config import input_data

# Importando a função feature_engineering_pipeline do arquivo indicadores.py
from indicadores import feature_engineering_pipeline

# Configuração do logging
logging.basicConfig(level=logging.DEBUG,  # DEBUG captura todos os logs
                    format='%(asctime)s - %(levelname)s - %(message)s')


def initialize_mt5():
    """Inicializa a conexão com o MetaTrader 5."""
    logging.info("Tentando conectar ao MetaTrader 5...")  # Log de tentativa de conexão
    if not mt5.initialize():
        logging.error("Erro ao conectar ao MetaTrader 5")  # Log de erro se a conexão falhar
        return False
    logging.info("Conexão bem-sucedida com o MetaTrader 5")  # Log de sucesso na conexão
    return True


def get_data(symbol, timeframe, num_candles):
    """Obtém dados do MetaTrader 5 para o símbolo e timeframe fornecidos."""

    # Verificação de validade do parâmetro 'symbol'
    if not isinstance(symbol, str) or len(symbol) == 0:
        logging.error("Símbolo inválido fornecido: %s", symbol)
        return None

    # Obtém os dados históricos do MetaTrader 5 usando a função copy_rates_from_pos.
    # A função retorna dados em formato de candles para o símbolo e timeframe fornecidos.
    data = mt5.copy_rates_from_pos(symbol, timeframe, 0, num_candles)

    # Verifica se a variável 'data' é None, o que indicaria que não foi possível obter os dados.
    # Nesse caso, um erro é registrado no log e a função retorna None.
    if data is None:
        logging.error("Erro ao obter dados para o símbolo %s", symbol)
        return None

    # Converte os dados de candles obtidos para um DataFrame usando o pandas.
    df = pd.DataFrame(data)

    # A coluna 'time' contém os timestamps dos candles, que são convertidos para o formato datetime.
    # 'unit="s"' indica que os valores estão em segundos desde a época Unix (1 de janeiro de 1970).
    df['time'] = pd.to_datetime(df['time'], unit='s')

    # Retorna o DataFrame contendo os dados históricos do símbolo, já com a coluna 'time' no formato datetime.
    return df


def add_tick_info(df, symbol):
    """Adiciona informações de tick (bid, ask e last_price) ao DataFrame."""

    # Validação do parâmetro 'symbol' para garantir que seja uma string não vazia
    if not isinstance(symbol, str) or len(symbol) == 0:
        logging.error("Símbolo inválido fornecido: %s", symbol)
        return df

    # Obtém as informações de tick (preço de bid e ask) do MetaTrader 5 para o símbolo fornecido
    tick = mt5.symbol_info_tick(symbol)

    # Verifica se as informações de tick foram obtidas com sucesso
    if tick is None:
        # Caso não seja possível obter as informações de tick, registra um aviso e retorna o DataFrame original sem alterações
        logging.warning("Informações de tick não disponíveis para %s", symbol)
        return df

    # Adiciona o preço de bid ao DataFrame
    df['bid'] = tick.bid

    # Adiciona o preço de ask ao DataFrame
    df['ask'] = tick.ask

    # Calcula o preço médio como a média entre bid e ask e adiciona ao DataFrame
    df['last_price'] = (df['bid'] + df['ask']) / 2

    # Retorna o DataFrame com as novas informações de tick
    return df


def calculate_market_returns(symbol, timeframe, num_candles):
    """Calcula os retornos logarítmicos do mercado."""

    # Obtém os dados do mercado usando a função 'get_data' (que retorna um DataFrame com candles)
    df_market = get_data(symbol, timeframe, num_candles)

    # Verifica se a função 'get_data' retornou um DataFrame válido e não vazio
    if df_market is None or df_market.empty:
        # Caso os dados do mercado não tenham sido obtidos com sucesso, registra um erro e retorna None
        logging.error("Não foi possível obter dados do mercado.")
        return None

    # Calcula os retornos logarítmicos do mercado:
    # O retorno logarítmico é calculado como a diferença entre o preço de fechamento de dois períodos consecutivos.
    # A fórmula é: log_ret = log(preço de fechamento atual / preço de fechamento anterior)
    df_market['log_ret'] = np.log(df_market['close'] / df_market['close'].shift(1))

    # Remove valores ausentes (NaN) que podem ser introduzidos ao calcular os retornos
    market_returns = df_market['log_ret'].dropna()

    # Retorna os retornos logarítmicos calculados, prontos para análise ou modelagem
    return market_returns


def get_last_period_data(symbol, timeframe, num_candles, frame_period):
    """Obtém os dados do último período e adiciona as métricas calculadas."""
    df = get_data(symbol, timeframe, num_candles)
    if df is None or df.empty:
        logging.error("Nenhum dado foi retornado.")
        return None

    # Criando uma cópia explícita do DataFrame para evitar a modificação de uma fatia
    df_last_period = df.tail(frame_period).copy()

    # Adicionando informações do tick
    df_last_period = add_tick_info(df_last_period, symbol)

    # Calculando os retornos do mercado
    market_returns = calculate_market_returns(symbol, timeframe, num_candles)

    # Certifique-se de que os retornos do mercado correspondem ao intervalo dos dados do último período
    market_returns_last_period = market_returns.tail(frame_period)  # Alinha os retornos com o período de interesse

    # Aplicando a função feature_engineering_pipeline para adicionar todas as métricas e indicadores
    try:
        df_last_period = feature_engineering_pipeline(df_last_period, symbol, market_returns_last_period)
    except Exception as e:
        logging.exception("Erro ao adicionar métricas no último período: %s", e)
        return None

    return df_last_period


def save_csv(df, symbol, timeframe, base_path):
    """Salva os dados calculados em um arquivo CSV dentro da pasta definida em config.py."""

    # Garante que o diretório 'input_data' exista. Se não existir, ele será criado.
    os.makedirs(input_data, exist_ok=True)  # Cria a pasta se não existir

    # Define o caminho completo e nome do arquivo CSV, baseado no símbolo, timeframe e pasta fornecida em 'input_data'.
    filename = os.path.join(input_data, f"{symbol}_{timeframe}_data.csv")

    # Salva o DataFrame em um arquivo CSV, sem incluir o índice e com codificação UTF-8.
    df.to_csv(filename, index=False, encoding='utf-8')

    # Exibe uma mensagem de confirmação no console indicando onde o arquivo foi salvo.
    print(f"✅ Arquivo salvo em: {filename}")

    # Tenta novamente salvar o arquivo CSV e registra o sucesso ou qualquer exceção que ocorrer.
    try:
        df.to_csv(filename, index=False, encoding="utf-8")
        # Se o arquivo for salvo com sucesso, registra uma mensagem de sucesso no log.
        logging.info("Arquivo CSV salvo com sucesso: %s", filename)

    except Exception as e:
        # Caso ocorra um erro durante o processo de salvamento, registra a exceção no log.
        logging.exception("Erro ao salvar CSV: %s", e)


def fetch_and_process_data(symbol, timeframe, num_candles, frame_period, base_path):
    """Função principal que obtém e processa os dados com base nos parâmetros fornecidos."""

    # Tenta inicializar a conexão com o MetaTrader 5 (MT5).
    if not initialize_mt5():
        return  # Se a conexão falhar, a função retorna e não prossegue.

    try:
        # Obtém os dados do último período com base no símbolo, timeframe e número de candles.
        df_last_period = get_last_period_data(symbol, timeframe, num_candles, frame_period)

        # Verifica se os dados do último período foram obtidos com sucesso e se não estão vazios.
        if df_last_period is None or df_last_period.empty:
            # Se não houver dados ou se o DataFrame estiver vazio, registra um erro e retorna.
            logging.error("Nenhum dado foi retornado para o último período.")
            return

        # Se os dados foram obtidos com sucesso, imprime uma mensagem de confirmação.
        print("Dados atualizados!")

        # Salva os dados calculados em um arquivo CSV.
        save_csv(df_last_period, symbol, timeframe, base_path)

        # Registra que as métricas do último período foram calculadas com sucesso.
        logging.info("Métricas do último período calculadas com sucesso.")

    except Exception as e:
        # Se ocorrer qualquer exceção durante o processamento, captura o erro e registra no log.
        logging.exception("Ocorreu um erro durante a execução: %s", e)

    finally:
        # Independentemente do sucesso ou falha, a função garante que a conexão com o MetaTrader 5 será encerrada.
        mt5.shutdown()
        logging.info("Conexão com o MetaTrader 5 encerrada.")
