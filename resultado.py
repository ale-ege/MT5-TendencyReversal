import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # ou o modelo que você estiver usando


def backtest(df, model_best, X_scaled, spread, valor_por_trade, alavancagem, capital_inicial):
    # Código do backtest, utilizando as variáveis passadas como parâmetros
    df['Predição'] = model_best.predict(X_scaled)

    # =============================================================================
    # 15. Aplicar o Modelo em Todo o Conjunto para o Backtest
    # =============================================================================
    print("Aplicando o modelo no conjunto completo para backtest...")
    df['Predição'] = model_best.predict(X_scaled)
    print("Modelo aplicado.\n")

    # =============================================================================
    # 16. Preparar o DataFrame para o Backtest (todas as linhas)
    # =============================================================================
    print("Preparando DataFrame para o backtest...")
    df["Estado"] = "Sem Operação"
    df["Entrada"] = np.nan
    df["Saída"] = np.nan
    df["Lucro/Prejuízo"] = 0.0
    df["Resultado (%)"] = 0.0
    df["Acumulado"] = np.nan  # Resultado acumulado de cada operação
    print("DataFrame preparado.\n")

    # Simulação Sequencial dos Trades com Fechamento Automático às 16:45:00
    print("Iniciando simulação do backtest...")
    trade_aberto = False
    trade_direcao = None   # "Long" ou "Short"
    preco_entrada = None
    num_acoes = None       # Número de ações compradas na operação
    trade_start_index = None

    for i in range(len(df)):
        if not trade_aberto:
            sinal = df.loc[i, "Predição"]
            if sinal == 1:
                trade_direcao = "Long"
                preco_entrada = df.loc[i, "close"] + spread
                num_acoes = (valor_por_trade * alavancagem) / preco_entrada
                trade_start_index = i
                df.loc[i, "Estado"] = "Compra"
                df.loc[i, "Entrada"] = preco_entrada
                df.loc[i, "Acumulado"] = 0.0
                trade_aberto = True
            elif sinal == -1:
                trade_direcao = "Short"
                preco_entrada = df.loc[i, "close"] - spread
                num_acoes = (valor_por_trade * alavancagem) / preco_entrada
                trade_start_index = i
                df.loc[i, "Estado"] = "Venda"
                df.loc[i, "Entrada"] = preco_entrada
                df.loc[i, "Acumulado"] = 0.0
                trade_aberto = True
            else:
                df.loc[i, "Estado"] = "Sem Operação"
        else:
            sinal = df.loc[i, "Predição"]
            if trade_direcao == "Long" and sinal == -1:
                preco_saida = df.loc[i, "close"] - spread
                resultado = (preco_saida - spread) - preco_entrada
                df.loc[i, "Estado"] = "Reversão Long"
                df.loc[i, "Saída"] = preco_saida
                df.loc[i, "Lucro/Prejuízo"] = num_acoes * resultado
                df.loc[i, "Acumulado"] = num_acoes * resultado
                trade_aberto = False
                trade_direcao = None
                preco_entrada = None
                num_acoes = None
                trade_start_index = None
            elif trade_direcao == "Short" and sinal == 1:
                preco_saida = df.loc[i, "close"] + spread
                resultado = preco_entrada - (preco_saida + spread)
                df.loc[i, "Estado"] = "Reversão Short"
                df.loc[i, "Saída"] = preco_saida
                df.loc[i, "Lucro/Prejuízo"] = num_acoes * resultado
                df.loc[i, "Acumulado"] = num_acoes * resultado
                trade_aberto = False
                trade_direcao = None
                preco_entrada = None
                num_acoes = None
                trade_start_index = None
            else:
                if trade_direcao == "Long":
                    acumulado = (df.loc[i, "close"] - spread - preco_entrada) * num_acoes
                    df.loc[i, "Estado"] = "Comprado"
                elif trade_direcao == "Short":
                    acumulado = (preco_entrada - (df.loc[i, "close"] + spread)) * num_acoes
                    df.loc[i, "Estado"] = "Vendido"
                df.loc[i, "Acumulado"] = acumulado

    print("Simulação do backtest concluída.\n")

    # =============================================================================
    # 17. Análise dos Resultados dos Últimos 30 Dias
    # =============================================================================
    print("Analisando resultados dos últimos 30 dias...")
    df['date'] = df['time'].dt.date
    # Agrupando resultados diários e contando transações
    daily_results = df.groupby('date').agg(
        total_profit_loss=('Lucro/Prejuízo', 'sum'),
        num_transactions=('Lucro/Prejuízo', 'size'),
        num_profit_transactions=('Lucro/Prejuízo', lambda x: (x > 0).sum()),
        num_loss_transactions=('Lucro/Prejuízo', lambda x: (x < 0).sum())
    )
    max_date = daily_results.index.max()
    last_30_days = daily_results[daily_results.index >= (max_date - pd.Timedelta(days=30))]
    last_30_days_df = last_30_days.reset_index()

    print("\n🔍 Análise dos Resultados dos Últimos 30 Dias:")
    print(last_30_days_df.to_string(index=False))
    total_30 = last_30_days['total_profit_loss'].sum()
    mean_30 = last_30_days['total_profit_loss'].mean()
    std_30 = daily_results['total_profit_loss'].std()  # Desvio padrão dos resultados diários
    print(f"\nTotal (30 dias): R$ {total_30:.2f}")
    print(f"Média Diária: R$ {mean_30:.2f}")
    print(f"Desvio Padrão: R$ {std_30:.2f}\n")

    # =============================================================================
    # 18. Cálculo da Média Mensal de Ganho
    # =============================================================================
    print("Calculando média mensal de ganho...")
    df['month'] = df['time'].dt.to_period('M')
    # Agrupando resultados mensais e contando transações
    monthly_results = df.groupby('month').agg(
        total_profit_loss=('Lucro/Prejuízo', 'sum'),
        num_transactions=('Lucro/Prejuízo', 'size'),
        num_profit_transactions=('Lucro/Prejuízo', lambda x: (x > 0).sum()),
        num_loss_transactions=('Lucro/Prejuízo', lambda x: (x < 0).sum())
    )
    monthly_mean = monthly_results['total_profit_loss'].mean()

    print("\n🔍 Média Mensal de Ganho:")
    print(monthly_results.reset_index().to_string(index=False))
    print(f"\nMédia Mensal: R$ {monthly_mean:.2f}\n")


    # =============================================================================
    # 25. Estatísticas das Operações Fechadas
    # =============================================================================
    print("Calculando estatísticas das operações fechadas...")
    fechadas = df[df["Estado"].isin(["Fechado Horário", "Reversão Long", "Reversão Short"])]
    qtd_profit = (fechadas["Lucro/Prejuízo"] > 0).sum()
    qtd_loss = (fechadas["Lucro/Prejuízo"] < 0).sum()
    soma_profit = fechadas.loc[fechadas["Lucro/Prejuízo"] > 0, "Lucro/Prejuízo"].sum()
    soma_loss = fechadas.loc[fechadas["Lucro/Prejuízo"] < 0, "Lucro/Prejuízo"].sum()
    max_gain = fechadas["Lucro/Prejuízo"].max()
    max_loss = fechadas["Lucro/Prejuízo"].min()

    print("\n🔍 Estatísticas das Operações Fechadas:")
    print(f"Quantidade de operações com lucro: {qtd_profit}")
    print(f"Quantidade de operações com prejuízo: {qtd_loss}")
    print(f"Somatório dos lucros: R$ {soma_profit:.2f}")
    print(f"Somatório dos prejuízos: R$ {soma_loss:.2f}")
    print(f"Ganho máximo em uma operação: R$ {max_gain:.2f}")
    print(f"Prejuízo máximo em uma operação: R$ {max_loss:.2f}\n")

    # =============================================================================
    # 26. Exibir Resumo Geral dos Resultados e Salvar CSV
    # =============================================================================
    print("Calculando resumo geral dos resultados...")
    total_lucro = df["Lucro/Prejuízo"].sum()
    roi = (total_lucro / capital_inicial) * 100
    capital_final = capital_inicial + total_lucro
    print(f"\n💰 Capital Inicial: R$ {capital_inicial:,.2f}")
    print(f"📈 Lucro/Prejuízo Total: R$ {total_lucro:.2f}")
    print(f"📊 ROI: {roi:.2f}%")
    print(f"🔹 Capital Final: R$ {capital_final:.2f}\n")
