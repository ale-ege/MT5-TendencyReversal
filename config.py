# config.py

# =============================================================================
# 1. Pastas e caminhos
# =============================================================================
# Caminho para salvar os treinamentos e dados
base_path = "./train"  # Defina o caminho para a pasta salvar os dados de treino
input_data = "./input_data"


# =============================================================================
# 2. Parâmetros treinamento
# =============================================================================
# Lista de ações a serem processadas
symbols = [
    'PETR4'
]


# =============================================================================
# 2. Parâmetros operacionais Treinamento
# =============================================================================
spread = 0.025          # Custo de transação %(entrada e saída)
valor_por_trade = 100   # Valor base investido por trade
alavancagem = 200       # Fator de alavancagem
capital_inicial = 1000  # Capital inicial
timeframe = mt5.TIMEFRAME_M5  # Time frame de operação
num_candles = 52500     # Quantidade de períodos analisados
frame_period = 52500    # Quantidade de períodos analisados
N = 9                   # Períodos válidos
threshold = 0.0035      # Ganho esperado %
importance_threshold = 0.0    # Define o limiar de importância para selecionar as features que têm importância maior que esse valor
