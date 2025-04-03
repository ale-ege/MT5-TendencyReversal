import os
import pytz
import time
import logging
import datetime
import joblib
import MetaTrader5 as mt5  # Importando o MetaTrader5

from bot_telegram import send_telegram_message, send_telegram_message_sinais
from Analisador import predict_trend_reversal  # Importando a função de predição

# Dados do Trade
symbol = os.path.splitext(os.path.basename(__file__))[0]
print(symbol)
timeframe = mt5.TIMEFRAME_M5  # Definindo o timeframe
num_candles = 75000
frame_period = 200

# Parâmetros operacionais
spread = 0.025  # Custo de transação (entrada e saída)
valor_por_trade = 10  # Valor base investido por trade
alavancagem = 200  # Fator de alavancagem
capital_inicial = 1000

# Horários de operação
hora_inicio = datetime.datetime.strptime("00:00:00", "%H:%M:%S").time()
hora_fim = datetime.datetime.strptime("19:55:00", "%H:%M:%S").time()

# Variáveis globais
ultimo_timestamp_salvo = None
novo_timestamp = None
trade_aberto = False
trade_direcao = None
preco_entrada = 0.0
num_acoes = 0.0
trade_start_index = None
estado_trade = None
preco_saida = 0.0
lucro_prejuizo = 0.0
acumulado = 0.0


# Função para enviar a mensagem de boas-vindas
def enviar_boas_vindas():
    hora_atual = datetime.datetime.now().strftime('%H:%M:%S')
    send_telegram_message(f"🚀🚀🚀  Olá, o bot foi iniciado com sucesso!💰💰💰💰💰\n 📉Papel: {symbol} 🕒 Horário: {hora_atual} \n")


# Função para criar a pasta de logs caso não exista
def criar_pasta_logs():
    if not os.path.exists('log'):
        os.makedirs('log')

# Configuração de logging
def configurar_logging():
    logging.basicConfig(filename='log/operacao.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Função para calcular o próximo ciclo de 5 minutos considerando o horário local
def calcular_proximo_ciclo():
    tz = pytz.timezone('America/Sao_Paulo')
    agora = datetime.datetime.now(tz)

    # Próximo ciclo é arredondado para cima, múltiplo de 5 minutos
    minutos_para_adicionar = 5 - (agora.minute % 5)
    proximo_ciclo = (agora + datetime.timedelta(minutes=minutos_para_adicionar)).replace(second=0, microsecond=0)

    # Calcula quantos segundos até o próximo ciclo
    tempo_ate_proximo_ciclo = (proximo_ciclo - agora).total_seconds()

    # Segurança para evitar tempo negativo
    tempo_ate_proximo_ciclo = max(tempo_ate_proximo_ciclo, 0)

    print(f"Aguardando até o próximo ciclo de 5 minutos: {proximo_ciclo.strftime('%H:%M:%S')}")
    return tempo_ate_proximo_ciclo



# Função principal para gerenciar o trade
def operacao():
    global trade_aberto, symbol, trade_direcao, preco_entrada, num_acoes, estado_trade, preco_saida, lucro_prejuizo, acumulado

    # Chama a função de predição para obter o último valor de abertura e a classe predita
    last_open_value, predicted_class = predict_trend_reversal(symbol, 5)
    sinal = predicted_class
    Valor_atual = last_open_value
    hora_atual = datetime.datetime.now().strftime('%H:%M:%S')
    print(f"✅ Papel: {symbol} 💸 Direção: {sinal} 📉 Cotação atual: {Valor_atual}🕒 Horário: {hora_atual} \n")
    send_telegram_message(f" 📉Papel: {symbol}  Direção: {sinal}  Cotação atual: {Valor_atual}🕒 Horário: {hora_atual} Trade: {estado_trade} \n")

    # Se não há trade aberto, tentamos abrir um novo
    if not trade_aberto:
        if sinal == 1:  # Compra
            trade_direcao = "Long"
            preco_entrada = Valor_atual + spread
            num_acoes = (valor_por_trade * alavancagem) / preco_entrada
            estado_trade = "Compra"
            print("Trade: Compra!")
            trade_aberto = True
            send_telegram_message_sinais(f"✅ Papel: {symbol}  Trade: {estado_trade}\n💸 Direção: {trade_direcao}\n📉 Preço de Entrada: {preco_entrada}")

        elif sinal == -1:  # Venda
            trade_direcao = "Short"
            preco_entrada = Valor_atual - spread
            num_acoes = (valor_por_trade * alavancagem) / preco_entrada
            estado_trade = "Venda"
            print("Trade: Venda!")
            trade_aberto = True
            send_telegram_message_sinais(f"✅ Papel: {symbol}  Trade: {estado_trade}\n💸 Direção: {trade_direcao}\n📉 Preço de Entrada: {preco_entrada}")

        else:  # Sem operação
            estado_trade = "Sem Operação"


    # Se o trade já está aberto, verificamos se ele precisa ser fechado ou revertido
    if trade_aberto:
        # Evite fazer a predição novamente caso o sinal seja "Sem Operação" e o trade não tenha mudado
        if sinal != 0:
            # Fechando o trade com base no horário ou sinal de reversão
            if datetime.datetime.now().time() == datetime.time(17, 55, 0):
                preco_saida = Valor_atual
                resultado = preco_saida - preco_entrada if trade_direcao == "Long" else preco_entrada - preco_saida
                estado_trade = "Fechado Horário"
                print("Trade: Fechado Horário!")
                lucro_prejuizo = num_acoes * resultado
                acumulado += lucro_prejuizo
                trade_aberto = False
                send_telegram_message_sinais(f"✅ Trade Fechado - {estado_trade}\n📈 Preço de Entrada: {preco_entrada}\n📉 Preço de Saída: {preco_saida}\n💰 Resultado: {lucro_prejuizo:.2f} R$\n📊 Acumulado: {acumulado:.2f} R$")

            elif (trade_direcao == "Long" and sinal == -1) or (trade_direcao == "Short" and sinal == 1):
                preco_saida = Valor_atual
                resultado = (preco_saida - preco_entrada) if trade_direcao == "Long" else (preco_entrada - preco_saida)
                estado_trade = "Reversão"
                print("Trade: Reversão!")
                lucro_prejuizo = num_acoes * resultado
                acumulado += lucro_prejuizo
                trade_aberto = False
                send_telegram_message_sinais(f"♻️ Trade Revertido - {estado_trade}\n📈 Preço de Entrada: {preco_entrada}\n📉 Preço de Saída: {preco_saida}\n💰 Resultado: {lucro_prejuizo:.2f} R$\n📊 Acumulado: {acumulado:.2f} R$")

            elif sinal == 0:
                if trade_direcao == "Long":
                    acumulado += (Valor_atual - preco_entrada) * num_acoes
                    estado_trade = "Comprado"
                    print("Trade: Comprado!")
                    send_telegram_message("Trade: Comprado!")
                elif trade_direcao == "Short":
                    acumulado += (preco_entrada - Valor_atual) * num_acoes
                    estado_trade = "Vendido"
                    print("Trade: Vendido!")
                    send_telegram_message("Trade: Vendido!")

        else:
            if trade_direcao == "Long":
                acumulado += (Valor_atual - preco_entrada) * num_acoes
                estado_trade = "Comprado"
            elif trade_direcao == "Short":
                acumulado += (preco_entrada - Valor_atual) * num_acoes
                estado_trade = "Vendido"

    return {
        "trade_aberto": trade_aberto,
        "trade_direcao": trade_direcao,
        "preco_entrada": preco_entrada,
        "num_acoes": num_acoes,
        "estado_trade": estado_trade,
        "preco_saida": preco_saida,
        "lucro_prejuizo": lucro_prejuizo,
        "acumulado": acumulado
    }


# Loop contínuo (24 horas por dia)
def loop_operacao():
    while True:
        hora_atual = datetime.datetime.now().time()

        if hora_inicio <= hora_atual <= hora_fim:
            tempo_espera = calcular_proximo_ciclo()
            print(f"Esperando {tempo_espera} segundos até o próximo ciclo...")
            time.sleep(tempo_espera)  # espera corretamente até o próximo ciclo
            time.sleep(1)
            operacao()
        else:
            logging.info("Fora do horário de operação. Aguardando...")
            print("Fora do horário de operação. Aguardando 60 segundos...")
            time.sleep(60)  # espera 1 minuto fora do horário de operação



# Executar o código
if __name__ == "__main__":
    enviar_boas_vindas()
    criar_pasta_logs()
    configurar_logging()
    loop_operacao()
