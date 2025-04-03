from treino import processar_acao

# =============================================================================
# 1. Variáveis internas
# =============================================================================

resultados_finais = []

# Loop principal
for symbol in symbols:
    try:
        # Inicia o processamento para cada símbolo da lista 'symbols'
        print(f"Processamento inciado: {symbol} !")

        # Chama a função 'processar_acao' para processar os dados do símbolo atual
        processar_acao(symbol)

        # Após o processamento ser concluído, imprime uma mensagem de sucesso
        print(f"Processamento Concluído: {symbol} !")

    except Exception as e:
        # Caso ocorra um erro ao processar o símbolo, captura a exceção e imprime uma mensagem de erro
        print(f"Erro ao processar {symbol}: {e}")

# Salvar Excel consolidado
excel_consolidado = "Resultados_Modelos.xlsx"

# Cria um DataFrame a partir da lista de resultados finais
df_resultados = pd.DataFrame(resultados_finais)

# Usa o ExcelWriter para salvar o DataFrame em um arquivo Excel
with pd.ExcelWriter(excel_consolidado, engine='xlsxwriter') as writer:
    # Salva os resultados na planilha "Resumo_Modelos" do arquivo Excel
    df_resultados.to_excel(writer, index=False, sheet_name="Resumo_Modelos")

# Após salvar o Excel, imprime uma mensagem indicando onde o arquivo foi salvo
print(f"\n✅ Resultados consolidados salvos em: {excel_consolidado}")
