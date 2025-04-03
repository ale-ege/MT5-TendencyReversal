import MetaTrader5 as mt5

def conectar_mt5():
    if not mt5.initialize():
        raise Exception(f"Erro ao iniciar MT5: {mt5.last_error()}")
    return True

def desconectar_mt5():
    mt5.shutdown()

def obter_lote_por_valor(symbol: str, valor: float):
    info = mt5.symbol_info(symbol)
    if not info:
        raise Exception(f"Informações do símbolo {symbol} não encontradas")
    preco = mt5.symbol_info_tick(symbol).ask
    volume = round(valor / preco, 2)  # arredonda para duas casas decimais
    if volume < info.volume_min:
        raise Exception(f"Volume calculado ({volume}) é menor que o mínimo permitido ({info.volume_min})")
    return volume

def comprar_ativo(symbol: str, valor: float):
    conectar_mt5()
    if not mt5.symbol_select(symbol, True):
        raise Exception(f"Não foi possível selecionar o símbolo {symbol}")

    tick = mt5.symbol_info_tick(symbol)
    volume = obter_lote_por_valor(symbol, valor)

    ordem = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": tick.ask,
        "deviation": 10,
        "magic": 1000,
        "comment": "compra via Python",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    resultado = mt5.order_send(ordem)
    desconectar_mt5()
    return resultado

def vender_ativo(symbol: str, valor: float):
    conectar_mt5()
    if not mt5.symbol_select(symbol, True):
        raise Exception(f"Não foi possível selecionar o símbolo {symbol}")

    tick = mt5.symbol_info_tick(symbol)
    volume = obter_lote_por_valor(symbol, valor)

    ordem = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL,
        "price": tick.bid,
        "deviation": 10,
        "magic": 1001,
        "comment": "venda via Python",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    resultado = mt5.order_send(ordem)
    desconectar_mt5()
    return resultado
