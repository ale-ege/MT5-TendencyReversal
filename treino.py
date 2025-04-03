import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import MetaTrader5 as mt5
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from config import symbols, spread, valor_por_trade, alavancagem, capital_inicial, timeframe, num_candles, frame_period, N, threshold, importance_threshold, base_path, input_data
from data_mt5 import fetch_and_process_data
from funcoes import plot_importance, testar_thresholds_importancia
import sys
import talib
import matplotlib
# Inicializando resultados finais
resultados_finais = []

print(talib.__version__)  # Deve retornar a versão (ex: 0.4.24)



def processar_acao(symbol, timeframe, num_candles=300, frame_period=300):
    """
    Processa os dados de um ativo financeiro específico, coleta dados do MetaTrader 5,
    cria diretórios para salvar os resultados e chama funções de processamento.

    Parâmetros:
        symbol (str): O símbolo do ativo financeiro (exemplo: 'PETR4', 'AAPL').
    """

    # =============================================================================
    # 1. Processamento
    # =============================================================================

    # Exibe uma mensagem indicando que o processamento do ativo foi iniciado.
    print(f"\nIniciando processamento de {symbol}...")

    # Backend não gráfico para evitar erros com Tkinter, importante para execução em servidores ou sem interface gráfica.

    matplotlib.use('Agg')  # Define o backend de matplotlib para não precisar de interface gráfica (útil em servidores sem GUI)
    import matplotlib.pyplot as plt  # Importa o matplotlib para criar gráficos (mesmo sem a GUI)

    # Configuração do Pandas para exibir todas as linhas de um DataFrame sem truncá-las.
    pd.set_option('display.max_rows', None)  # Define que o Pandas deve exibir todas as linhas de um DataFrame

    # Define o caminho da pasta Train e cria o diretório para o símbolo e timeframe
    symbol_timeframe_path = os.path.join(base_path, f"{symbol}_{timeframe}")

    # Verifica se o diretório já existe, caso contrário, cria-o
    if not os.path.exists(symbol_timeframe_path):
        os.makedirs(symbol_timeframe_path)
        print(f"Diretório criado para {symbol} e timeframe {timeframe}")

    # Chama a função 'fetch_and_process_data' para coletar e processar os dados financeiros do ativo
    fetch_and_process_data(symbol, timeframe, num_candles, frame_period, base_path)


    # =============================================================================
    # 2. Carregar os Dados Originais df
    # =============================================================================
    print("Carregando dados do arquivo...")

    # Define o caminho para o arquivo de dados usando a variável input_data
    file_original = os.path.join(input_data, f"{symbol}_{timeframe}_data.csv")

    # Verifica se o arquivo existe e carrega os dados
    if os.path.exists(file_original):
        df = pd.read_csv(file_original)
        print(f"Dados carregados: {df.shape[0]} linhas, {df.shape[1]} colunas.\n")
    else:
        print(f"Erro: O arquivo {file_original} não foi encontrado.")
        exit()


    # =============================================================================
    # 3. Relatório de dados
    # =============================================================================

    # Salvar a saída padrão original para restaurar posteriormente
    original_stdout = sys.stdout

    # Caminho do arquivo onde a saída será salva, usando o caminho do símbolo e timeframe
    log_file_path = os.path.join(symbol_timeframe_path, f"{symbol}_{timeframe}_relatorio.doc")

    # Abre o arquivo .doc para gravação. Se o arquivo não existir, será criado.
    log_file = open(log_file_path, 'w', encoding='utf-8')

    # Redireciona a saída padrão para o arquivo de log, para que tudo o que for impresso seja gravado no arquivo
    sys.stdout = log_file

    # Impressões no arquivo de relatório com informações sobre o ativo e timeframe
    print(f"RELATÓRIO CRIADO:  {symbol}\n")
    print(f"TIMEFRAME:  {timeframe}\n")


    # =============================================================================
    # 4. Converter 'time' para datetime
    # =============================================================================

    # Converte a coluna 'time' do DataFrame de timestamps (em formato Unix) para o formato datetime
    df['time'] = pd.to_datetime(df['time'])

    # Exibe uma mensagem indicando que a conversão foi concluída com sucesso
    print("Conversão da coluna 'time' para datetime concluída!\n")


    # =============================================================================
    # 5. Gerar o Rótulo (Target) para Reversão de Tendência
    # =============================================================================

    # Função para verificar a existência de uma coluna no DataFrame
    def check_column_exists(df, column_name):
        # Verifica se a coluna especificada existe no DataFrame
        if column_name not in df.columns:
            # Exibe uma mensagem de erro caso a coluna não seja encontrada
            print(f"Erro: A coluna '{column_name}' não foi encontrada no DataFrame.")
            return False  # Retorna False caso a coluna não exista
        return True  # Retorna True caso a coluna exista

    # Verificação antes de acessar a coluna 'target'
    if not check_column_exists(df, 'target'):
        # Caso a coluna 'target' não exista, ela será criada
        print("Criando a coluna 'target' para reversão de tendência...")

        # Calcula o retorno futuro, deslocando os preços de fechamento para o futuro (N períodos)
        df['future_return'] = df['close'].shift(-N) / df['close'] - 1

        # Calcula o retorno atual (percentual de mudança)
        df['current_return'] = df['close'].pct_change()

        # Cria a coluna 'target' com valores iniciais de 0 (neutro)
        df['target'] = 0

        # Define o rótulo como 1 para reversão de tendência para cima (compra)
        # quando o retorno atual é negativo e o retorno futuro é maior que o threshold
        df.loc[(df['current_return'] < 0) & (df['future_return'] > threshold), 'target'] = 1

        # Define o rótulo como -1 para reversão de tendência para baixo (venda)
        # quando o retorno atual é positivo e o retorno futuro é menor que o threshold negativo
        df.loc[(df['current_return'] > 0) & (df['future_return'] < -threshold), 'target'] = -1

        # Remove as últimas N linhas para evitar dados incompletos ao usar o deslocamento (shift)
        df = df.iloc[:-N]

        # Reseta o índice do DataFrame para manter a consistência após o corte das linhas
        df.reset_index(drop=True, inplace=True)

        # Exibe uma mensagem informando que a coluna 'target' foi criada
        print("Coluna 'target' criada.\n")

    # Agora a coluna 'target' pode ser acessada com segurança
    y = df['target']


    # =============================================================================
    # 6. Definir as Features para o Modelo
    # =============================================================================

    # Lista com os nomes de todas as possíveis features que podem ser usadas no modelo
    features = [
        '+DM', '+DM_sum', '-DM', '-DM_sum', 'adl', 'adf_stat_30', 'ask', 'atr_14', 'atr_50', 'avg_price', 'bear_power',
        'beta', 'bid', 'bollinger_percent_b', 'bull_power', 'cci_14', 'change_percent', 'cmf_20', 'close',
        'close_lag1', 'close_lag2', 'close_lag3', 'cum_max', 'drawdown', 'ema_13', 'ema_20', 'ewma_30', 'high', 'high_lag1',
        'high_lag2', 'high_lag3', 'hma_14', 'hma_50', 'last_price', 'low', 'low_lag1', 'low_lag2', 'low_lag3', 'log_ret', 'macd',
        'macd_hist', 'macd_signal', 'momentum_10', 'mfi_10', 'open', 'open_lag1', 'open_lag2', 'open_lag3', 'obv', 'roc_20',
        'roc_5', 'rsi_14', 'sharpe_ratio', 'sma_100', 'sma_200', 'sma_21', 'sma_50', 'sma_9', 'sma_cross_signal_50_200',
        'sma_cross_signal_9_21', 'sma_diff_50_200', 'sma_diff_9_21', 'spread', 'stoch_d', 'stoch_k', 'tick_volume',
        'TR_sum', 'ulcer_index', 'vwap_20', 'volume_profile', 'vw_rsi', 'vroc_14', 'williams_r_14', 'zscore_vol'
    ]

    # Filtra as features disponíveis no DataFrame, garantindo que as features que existem no DataFrame sejam selecionadas
    available_features = [feat for feat in features if feat in df.columns]

    # Seleciona as colunas do DataFrame que correspondem às features disponíveis
    X = df[available_features]

    # Define a coluna 'target' como a variável dependente (rótulo) para o modelo
    y = df['target']

    # Exibe as features definidas que foram encontradas no DataFrame
    print(f"Features definidas: {available_features}\n")


    # =============================================================================
    # 7. Imputação dos Valores Ausentes e Normalização
    # =============================================================================

    # Cria um objeto SimpleImputer para imputar valores ausentes utilizando a média
    imputer = SimpleImputer(strategy='mean')

    # Aplica a imputação dos valores ausentes no conjunto de features X
    # O método fit_transform ajusta o imputer aos dados e aplica a transformação, substituindo valores ausentes pela média
    X_imputed = imputer.fit_transform(X)

    # Cria um objeto StandardScaler para normalizar os dados
    scaler = StandardScaler()

    # Aplica a normalização (padronização) nos dados imputados, transformando para ter média 0 e desvio padrão 1
    X_scaled = scaler.fit_transform(X_imputed)

    # Exibe uma mensagem informando que a imputação e a normalização foram concluídas com sucesso
    print("Imputação e normalização concluídas!\n")


    # =============================================================================
    # 8. Dividir em Treino e Teste
    # =============================================================================

    # Utiliza a função train_test_split do scikit-learn para dividir os dados em conjuntos de treino e teste
    # X_scaled: As features normalizadas
    # y: O alvo (target) a ser previsto
    # test_size=0.7: Define que 70% dos dados serão usados para o conjunto de teste e 30% para treino
    # random_state=42: Garante que a divisão dos dados seja reprodutível, ou seja, os mesmos dados de treino e teste serão gerados em cada execução
    # stratify=y: Garante que a divisão entre treino e teste preserve a distribuição da variável alvo (y) nos dois conjuntos
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.7, random_state=42, stratify=y
    )

    # Exibe uma mensagem informando que os dados foram divididos corretamente em treino e teste
    print("Dados divididos em treino e teste!\n")


    # =============================================================================
    # 9. Modelo RandomForest e Avaliação
    # =============================================================================

    # Exibe uma mensagem informando que o modelo RandomForest está sendo treinado
    print("Treinando modelo RandomForest...")

    # Criação do modelo RandomForestClassifier com parâmetros específicos
    # n_estimators=100: Define o número de árvores na floresta. Quanto maior esse número, mais preciso o modelo pode ser, mas também mais computacionalmente caro.
    # max_depth=None: Não limita a profundidade das árvores. Isso permite que as árvores cresçam sem restrições, o que pode aumentar a complexidade do modelo.
    # random_state=42: Garante que o modelo seja reprodutível, ou seja, os mesmos resultados serão gerados em cada execução
    # n_jobs=-1: Utiliza todos os núcleos do processador para treinar o modelo, acelerando o processo
    model_best = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)

    # Treina o modelo com os dados de treino (X_train e y_train)
    model_best.fit(X_train, y_train)

    # Faz a previsão com os dados de teste
    y_pred_test = model_best.predict(X_test)

    # Avalia o modelo utilizando a acurácia, comparando as previsões (y_pred_test) com os valores reais (y_test)
    accuracy = accuracy_score(y_test, y_pred_test)

    # Exibe a acurácia do modelo, mostrando o desempenho do modelo no conjunto de dados de teste
    print(f"\nAcurácia do Modelo: {accuracy:.2%}\n")

    # =============================================================================
    # 10. Análise de Importância das Features
    # =============================================================================

    # Obtém a importância das features do modelo RandomForest treinado
    # O atributo 'feature_importances_' do modelo RandomForest retorna a importância de cada feature
    importances = model_best.feature_importances_

    # Cria um objeto pd.Series a partir das importâncias das features e ordena em ordem decrescente
    # O índice do pd.Series é as features disponíveis que foram usadas no modelo
    feat_importance = pd.Series(importances, index=available_features).sort_values(ascending=False)

    # Gera um gráfico de barras para visualizar a importância das features
    # 'kind='bar'' especifica que o gráfico será do tipo barra, e 'figsize=(10, 6)' define o tamanho da figura
    feat_importance.plot(kind='bar', title='Importância das Features', figsize=(10, 6))

    # Define o caminho para salvar o gráfico de importância das features em formato PNG
    # O caminho é baseado no símbolo e timeframe, e o nome do arquivo inclui o sufixo '_feature_importance_sorted_by_value'
    plot_filename = os.path.join(symbol_timeframe_path, f"{symbol}_{timeframe}_feature_importance_RandomForest_treinado.png")

    # Salva o gráfico gerado no arquivo especificado
    plt.savefig(plot_filename)

    # Exibe uma mensagem informando o local onde o gráfico foi salvo
    print(f"Gráfico de Importância das Features (ordenado por valor) salvo em: {plot_filename}")

    # Fecha o gráfico após salvar para liberar memória
    plt.close()


    # =============================================================================
    # 11 Função Testar Melhor Thresholds
    # =============================================================================



        # Converte os resultados para um DataFrame
        df_resultados = pd.DataFrame(resultados, columns=["Threshold", "N_Features", "Accuracy"])

        # Exibe os resultados no console
        print("\n🔍 Comparação de Acurácia por Threshold:")
        print(df_resultados.to_string(index=False))

        # Gera o gráfico de comparação entre thresholds e acurácia
        plt.figure(figsize=(10, 6))
        plt.plot(df_resultados["Threshold"], df_resultados["Accuracy"], marker='o')
        plt.title("Variação da Acurácia com Thresholds de Importância")
        plt.xlabel("Threshold de Importância")
        plt.ylabel("Acurácia")
        plt.grid(True)

        # Define o caminho para salvar o gráfico
        plot_filename = os.path.join(symbol_timeframe_path, f"{symbol}_{timeframe}_threshold_vs_accuracy.png")
        plt.savefig(plot_filename)

        # Exibe o caminho onde o gráfico foi salvo
        print(f"Gráfico de análise de thresholds salvo em: {plot_filename}")
        plt.close()

        # Retorna o DataFrame com os resultados
        return df_resultados


    # =============================================================================
    # 12. Análise de Permutation Importance das Features e Seleção das Relevantes
    # =============================================================================

    # Criar um DataFrame com as features nomeadas para análise, a partir dos dados normalizados
    X_df = pd.DataFrame(X_scaled, columns=available_features)

    # Rodar a análise comparativa de thresholds (definido anteriormente)
    # Isso testa o impacto de diferentes thresholds de importância na acurácia do modelo
    testar_thresholds_importancia(model_best, X_df, y, symbol_timeframe_path, symbol, timeframe)

    # Continuar com a análise de Permutation Importance para determinar as features mais relevantes
    # Exibe uma mensagem indicando o início da análise de Permutation Importance
    print("Analisando Permutation Importance das Features (para seleção final)...")

    # Calcula a Permutation Importance no conjunto de teste usando o modelo RandomForest
    # n_repeats=10 faz 10 permutações para calcular a importância média de cada feature
    # random_state=42 garante que os resultados sejam reproduzíveis
    perm_result = permutation_importance(model_best, X_test, y_test, n_repeats=10, random_state=42)

    # Atribui a importância das features calculada pela permutação em um pd.Series
    # index=available_features assegura que cada valor de importância seja associado à sua respectiva feature
    perm_importance = pd.Series(perm_result.importances_mean, index=available_features).sort_values(ascending=False)

    # Exibe a Permutation Importance das features no console
    # As features com maior importância serão exibidas no topo
    print("\n🔍 Permutation Importance das Features:")
    print(perm_importance)

    # Gera um gráfico de barras para visualizar a Permutation Importance das features
    # 'kind='bar'' especifica que o gráfico será do tipo barra, e 'figsize=(10, 6)' define o tamanho do gráfico
    perm_importance.plot(kind='bar', title='Permutation Importance das Features', figsize=(10, 6))

    # Define o caminho para salvar o gráfico gerado
    plot_filename_perm = os.path.join(symbol_timeframe_path, f"{symbol}_{timeframe}_permutation_importance_mean_sorted.png")

    # Salva o gráfico de Permutation Importance como uma imagem PNG
    plt.savefig(plot_filename_perm)

    # Exibe uma mensagem informando onde o gráfico foi salvo
    print(f"Gráfico de Permutation Importance das Features (média ordenada) salvo em: {plot_filename_perm}")
    plt.close()  # Fecha o gráfico para liberar memória

    # Seleciona as features cuja importância é maior que o limiar (threshold)
    selected_features = perm_importance[perm_importance > importance_threshold].index.tolist()

    # Seleciona os dados de entrada (features) com base nas features selecionadas
    X_selected = df[selected_features]

    # Atualiza o dataset com as features selecionadas, imputando valores ausentes e normalizando os dados
    imputer = SimpleImputer(strategy='mean')  # Imputa valores ausentes usando a média
    X_imputed = imputer.fit_transform(X_selected)  # Aplica a imputação
    scaler = StandardScaler()  # Cria o normalizador
    X_scaled = scaler.fit_transform(X_imputed)  # Normaliza os dados

    # Divide os dados em conjuntos de treino e teste
    # test_size=0.3 define que 30% dos dados irão para o conjunto de teste
    # random_state=42 garante que a divisão seja reprodutível
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Salva as features selecionadas para uso futuro, armazenando-as em um arquivo pickle
    selected_features_path = os.path.join(symbol_timeframe_path, f'{symbol}_{timeframe}_selected_features.pkl')
    joblib.dump(selected_features, selected_features_path)

    # Exibe uma mensagem indicando que as features selecionadas foram salvas com sucesso
    print(f"Features selecionadas salvas em '{selected_features_path}'.")

    # Re-treina o modelo RandomForest usando as features selecionadas
    print("\nTreinando modelo RandomForest com features selecionadas...")
    model_best = RandomForestClassifier(n_estimators=100, max_depth=None, random_state=42, n_jobs=-1)
    model_best.fit(X_train, y_train)  # Treina o modelo com os dados de treino

    # Avalia o modelo no conjunto de teste e calcula a acurácia
    accuracy = accuracy_score(y_test, model_best.predict(X_test))

    # Exibe a nova acurácia do modelo após a seleção das features
    print(f"\n🔍 Nova Acurácia do Modelo com Features Selecionadas: {accuracy:.2%}\n")


    # =============================================================================
    # 13 Ajuste Fino dos Hiperparâmetros com GridSearchCV
    # =============================================================================

    # Exibe uma mensagem informando o início da busca pelos melhores hiperparâmetros
    # A busca é feita utilizando a técnica de ajuste fino de parâmetros com o GridSearchCV ou RandomizedSearchCV
    print("Iniciando busca pelos melhores hiperparâmetros com GridSearchCV...")

    # Definindo a grade de parâmetros a ser explorada
    # A grade contém uma lista de valores para os hiperparâmetros do modelo RandomForest que serão testados
    # O número de combinações é reduzido para evitar a exploração excessiva do espaço de parâmetros

    tuned_param_grid = {
        'n_estimators': [100, 200, 300],  # Número de árvores na floresta
        'max_depth': [None, 10, 15, 20],  # Profundidade máxima das árvores
        'min_samples_split': [2, 5, 10],  # Número mínimo de amostras necessárias para dividir um nó
        'min_samples_leaf': [1, 2],  # Número mínimo de amostras necessárias para ser uma folha
        'max_features': ['sqrt', 'log2'],  # Número máximo de features a serem consideradas para dividir um nó
        'bootstrap': [True]  # Usar amostragem bootstrap para treinar as árvores
    }

    """
    Comentário sobre os melhores parâmetros encontrados:
    Os melhores parâmetros encontrados com base em uma busca anterior poderiam ser:
    {'n_estimators': 100, 'min_samples_split': 2, 'min_samples_leaf': 1, 
     'max_features': 'sqrt', 'max_depth': None, 'bootstrap': True}
    """

    # Instancia o modelo RandomForestClassifier, que será ajustado com os melhores parâmetros
    # O parâmetro random_state=42 garante que os resultados sejam reprodutíveis
    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    # RandomizedSearchCV permite explorar aleatoriamente um número específico de combinações de parâmetros
    # n_iter=50 define o número de combinações aleatórias de parâmetros que o RandomizedSearchCV testará
    # cv=3 indica que será usado 3-fold cross-validation para avaliar cada combinação
    # verbose=2 mostra detalhes sobre o progresso da busca
    # n_jobs=-1 utiliza todos os núcleos do processador para acelerar a busca
    random_search = RandomizedSearchCV(
        estimator=model,  # O modelo a ser ajustado
        param_distributions=tuned_param_grid,  # A grade de parâmetros a ser testada
        n_iter=50,  # Número de combinações aleatórias a serem testadas
        cv=3,  # Número de folds para cross-validation
        verbose=2,  # Nível de detalhes sobre o progresso da busca
        random_state=42,  # Garante a reprodutibilidade da busca
        n_jobs=-1  # Utiliza todos os núcleos do processador
    )

    # Ajusta o modelo aos dados de treino, buscando os melhores parâmetros
    random_search.fit(X_train, y_train)

    # Exibe os melhores parâmetros encontrados durante a busca
    # O método 'best_params_' retorna o dicionário com os melhores parâmetros encontrados
    print("\nMelhores parâmetros encontrados:")
    print(random_search.best_params_)

    # Atualiza o modelo com os melhores parâmetros encontrados
    # O modelo com os melhores parâmetros é acessado através do atributo 'best_estimator_'
    model_best = random_search.best_estimator_

    # Avaliação com os melhores hiperparâmetros encontrados
    # Utiliza o modelo ajustado com os melhores parâmetros para fazer previsões no conjunto de teste
    y_pred_test = model_best.predict(X_test)

    # Calcula a acurácia do modelo nas previsões feitas com os dados de teste
    accuracy = accuracy_score(y_test, y_pred_test)

    # Exibe a acurácia do modelo utilizando os melhores hiperparâmetros
    print(f"\n\U0001F50D Acurácia do Modelo (Teste) com melhores hiperparâmetros: {accuracy:.2%}\n")


    # =============================================================================
    # 14. Salvar o Melhor Modelo, Imputer e Scaler
    # =============================================================================

    # O objetivo desta etapa é salvar o modelo treinado, o imputer (para lidar com valores ausentes)
    # e o scaler (para normalização dos dados) em arquivos para uso futuro. Esses arquivos são
    # salvos no diretório específico para o símbolo e o timeframe, para facilitar a reutilização
    # do modelo e das transformações em futuras previsões.

    # Salva o modelo treinado (modelo com os melhores parâmetros encontrados) em um arquivo .pkl
    # O arquivo será armazenado na pasta referente ao símbolo e timeframe, com o nome 'best_model_grid_search.pkl'
    joblib.dump(model_best, os.path.join(symbol_timeframe_path, f'{symbol}_{timeframe}_best_model_grid_search.pkl'))

    # Exibe uma mensagem indicando onde o modelo foi salvo
    print(f"Modelo salvo em '{symbol_timeframe_path}/{symbol}_{timeframe}_best_model_grid_search.pkl'.")

    # Salva o imputer (usado para preencher valores ausentes com a média, no caso) em um arquivo .pkl
    # O arquivo será armazenado na mesma pasta, com o nome 'imputer.pkl'
    joblib.dump(imputer, os.path.join(symbol_timeframe_path, f'{symbol}_{timeframe}_imputer.pkl'))

    # Exibe uma mensagem indicando onde o imputer foi salvo
    print(f"Imputer salvo em '{symbol_timeframe_path}/{symbol}_{timeframe}_imputer.pkl'.")

    # Salva o scaler (usado para normalizar os dados) em um arquivo .pkl
    # O arquivo será armazenado na mesma pasta, com o nome 'scaler.pkl'
    joblib.dump(scaler, os.path.join(symbol_timeframe_path, f'{symbol}_{timeframe}_scaler.pkl'))

    # Exibe uma mensagem indicando onde o scaler foi salvo
    print(f"Scaler salvo em '{symbol_timeframe_path}/{symbol}_{timeframe}_scaler.pkl'.\n")


    # =============================================================================
    # 15. Análise de Importância das Features
    # =============================================================================

    # Inicia a análise de importância das features utilizando o modelo RandomForest
    print("Analisando importância das features...")

    # Acessa a importância das features do modelo RandomForest
    # O atributo 'feature_importances_' retorna a importância de cada feature para o modelo
    importances = model_best.feature_importances_

    # Cria um pd.Series com as importâncias, associando a cada feature sua respectiva importância
    # O resultado é ordenado em ordem decrescente para facilitar a visualização das features mais importantes
    feat_importance = pd.Series(importances, index=selected_features).sort_values(ascending=False)

    # Exibe as importâncias das features no console
    print("\n🔍 Importância das Features (RandomForest):")
    print(feat_importance)

    # Gera um gráfico de barras para visualizar a importância das features
    # 'kind='bar'' especifica que o gráfico será do tipo barra
    # 'figsize=(10, 6)' define o tamanho do gráfico
    feat_importance.plot(kind='bar', title='Importância das Features', figsize=(10, 6))

    # Define o caminho para salvar o gráfico na pasta "Train", organizando os gráficos por símbolo e timeframe
    symbol_timeframe_path = os.path.join(base_path, f"{symbol}_{timeframe}")  # Caminho para o diretório do símbolo e timeframe

    # Verifica se a pasta para salvar o gráfico existe, se não, cria a pasta
    if not os.path.exists(symbol_timeframe_path):
        os.makedirs(symbol_timeframe_path)

    # Define o nome do arquivo onde o gráfico será salvo
    plot_filename = os.path.join(symbol_timeframe_path, f"{symbol}_{timeframe}_feature_importance2.png")

    # Salva o gráfico de importância das features como uma imagem PNG
    plt.savefig(plot_filename)
    print(f"Gráfico de Importância das Features salvo em: {plot_filename}")

    # Fecha o gráfico para liberar os recursos de memória utilizados na criação do gráfico
    plt.close()


    # =============================================================================
    # 16. Análise de Permutation Importance das Features
    # =============================================================================
    # Realiza a análise de Permutation Importance das features
    # A permutation importance verifica o impacto da permutação de uma feature na performance do modelo
    perm_result = permutation_importance(model_best, X_test, y_test, n_repeats=10, random_state=42)

    # Cria um pd.Series com a média da importância das features, ordenando por importância em ordem decrescente
    perm_importance = pd.Series(perm_result.importances_mean, index=selected_features).sort_values(ascending=False)

    # Exibe a Permutation Importance das features no console
    print("\n🔍 Permutation Importance das Features:")
    print(perm_importance)

    # Gera um gráfico de barras para visualizar a Permutation Importance das features
    perm_importance.plot(kind='bar', title='Permutation Importance das Features', figsize=(10, 6))

    # Define o caminho para salvar o gráfico da Permutation Importance
    # O gráfico será salvo na mesma pasta "Train", com o nome do arquivo específico para Permutation Importance
    plot_filename_perm = os.path.join(symbol_timeframe_path, f"{symbol}_{timeframe}_permutation_importance3.png")

    # Salva o gráfico de Permutation Importance das features como uma imagem PNG
    plt.savefig(plot_filename_perm)
    print(f"Gráfico de Permutation Importance das Features salvo em: {plot_filename_perm}")

    # Fecha o gráfico para liberar os recursos de memória utilizados na criação do gráfico
    plt.close()


    # =============================================================================
    # 17. Aplicar o Modelo em Todo o Conjunto para o Backtest
    # =============================================================================
    # Aplica o modelo treinado (model_best) no conjunto completo de dados (X_scaled)
    # A variável X_scaled contém as features de entrada, que já foram normalizadas e preparadas
    # O modelo realiza previsões para cada linha do conjunto de dados
    print("Aplicando o modelo no conjunto completo para backtest...")

    # A coluna 'Predição' é adicionada ao DataFrame 'df', com as previsões do modelo
    # Cada linha do DataFrame recebe um valor de previsão (-1, 0, ou 1) com base no modelo treinado
    df['Predição'] = model_best.predict(X_scaled)

    # Mensagem indicando que a aplicação do modelo foi concluída
    print("Modelo aplicado.\n")


    # =============================================================================
    # 18. Preparar o DataFrame para o Backtest (todas as linhas)
    # =============================================================================
    # Exibe uma mensagem indicando que a preparação do DataFrame para o backtest está em andamento
    print("Preparando DataFrame para o backtest...")

    # Inicializa as colunas do DataFrame com valores padrão
    # "Estado" indica o status da operação (compra, venda, sem operação)
    df["Estado"] = "Sem Operação"  # Inicialmente, não há operação aberta
    df["Entrada"] = np.nan  # Preço de entrada será definido mais tarde
    df["Saída"] = np.nan  # Preço de saída será definido durante o fechamento da operação
    df["Lucro/Prejuízo"] = 0.0  # Lucro ou prejuízo de cada operação será calculado durante o backtest
    df["Resultado (%)"] = 0.0  # Resultado percentual de cada operação
    df["Acumulado"] = np.nan  # Resultado acumulado de cada operação

    # Confirma a preparação do DataFrame
    print("DataFrame preparado.\n")

    # =============================================================================
    # Simulação Sequencial dos Trades com Fechamento Automático às 16:45:00
    # =============================================================================
    # Exibe a mensagem indicando que a simulação do backtest está começando
    print("Iniciando simulação do backtest...")

    # Variáveis para controlar o estado da operação de trade
    trade_aberto = False  # Inicia com nenhuma operação aberta
    trade_direcao = None  # Pode ser "Long" ou "Short", define a direção do trade
    preco_entrada = None  # O preço de entrada será atribuído quando uma operação for aberta
    num_acoes = None  # Número de ações compradas ou vendidas
    trade_start_index = None  # O índice onde o trade começou no DataFrame

    # Loop que percorre todas as linhas do DataFrame para simular os trades
    for i in range(len(df)):
        # Se não houver operação aberta, verifica o sinal da previsão (coluna 'Predição')
        if not trade_aberto:
            sinal = df.loc[i, "Predição"]  # Verifica o sinal gerado pelo modelo
            if sinal == 1:
                # Sinal de compra ("Long"), define as variáveis e marca o início da operação
                trade_direcao = "Long"
                preco_entrada = df.loc[i, "close"] + spread  # Preço de entrada com o spread
                num_acoes = (valor_por_trade * alavancagem) / preco_entrada  # Cálculo de quantas ações comprar
                trade_start_index = i  # Salva o índice de início do trade
                df.loc[i, "Estado"] = "Compra"  # Marca a operação como "Compra"
                df.loc[i, "Entrada"] = preco_entrada  # Registra o preço de entrada
                df.loc[i, "Acumulado"] = 0.0  # Inicializa o resultado acumulado
                trade_aberto = True  # Indica que a operação foi aberta
            elif sinal == -1:
                # Sinal de venda ("Short"), define as variáveis e marca o início da operação
                trade_direcao = "Short"
                preco_entrada = df.loc[i, "close"] - spread  # Preço de entrada com o spread
                num_acoes = (valor_por_trade * alavancagem) / preco_entrada  # Cálculo de quantas ações vender
                trade_start_index = i  # Salva o índice de início do trade
                df.loc[i, "Estado"] = "Venda"  # Marca a operação como "Venda"
                df.loc[i, "Entrada"] = preco_entrada  # Registra o preço de entrada
                df.loc[i, "Acumulado"] = 0.0  # Inicializa o resultado acumulado
                trade_aberto = True  # Indica que a operação foi aberta
            else:
                # Caso não haja operação (sinal == 0), marca como "Sem Operação"
                df.loc[i, "Estado"] = "Sem Operação"
        else:
            # Se houver operação aberta, verifica o sinal da previsão para encerrar a operação
            sinal = df.loc[i, "Predição"]
            if trade_direcao == "Long" and sinal == -1:
                # Se for uma posição Long e o sinal for de venda, fecha a operação
            preco_saida = df.loc[i, "close"] - spread  # Preço de saída com o spread
            resultado = (preco_saida - spread) - preco_entrada  # Cálculo do resultado
            df.loc[i, "Estado"] = "Reversão Long"  # Marca como reversão de longo
            df.loc[i, "Saída"] = preco_saida  # Registra o preço de saída
            df.loc[i, "Lucro/Prejuízo"] = num_acoes * resultado  # Calcula lucro/prejuízo
            df.loc[i, "Acumulado"] = num_acoes * resultado  # Resultado acumulado
            trade_aberto = False  # Fecha o trade
            trade_direcao = None  # Zera a direção do trade
            preco_entrada = None  # Zera o preço de entrada
            num_acoes = None  # Zera o número de ações
            trade_start_index = None  # Zera o índice de início
        elif trade_direcao == "Short" and sinal == 1:
            # Se for uma posição Short e o sinal for de compra, fecha a operação
            preco_saida = df.loc[i, "close"] + spread  # Preço de saída com o spread
            resultado = preco_entrada - (preco_saida + spread)  # Cálculo do resultado
            df.loc[i, "Estado"] = "Reversão Short"  # Marca como reversão de curto
            df.loc[i, "Saída"] = preco_saida  # Registra o preço de saída
            df.loc[i, "Lucro/Prejuízo"] = num_acoes * resultado  # Calcula lucro/prejuízo
            df.loc[i, "Acumulado"] = num_acoes * resultado  # Resultado acumulado
            trade_aberto = False  # Fecha o trade
            trade_direcao = None  # Zera a direção do trade
            preco_entrada = None  # Zera o preço de entrada
            num_acoes = None  # Zera o número de ações
            trade_start_index = None  # Zera o índice de início
        else:
            # Caso o sinal não seja de reversão, mantém o trade aberto e calcula o valor acumulado
            if trade_direcao == "Long":
                acumulado = (df.loc[i, "close"] - spread - preco_entrada) * num_acoes  # Cálculo do acumulado para Long
                df.loc[i, "Estado"] = "Comprado"  # Marca a operação como "Comprado"
            elif trade_direcao == "Short":
                acumulado = (preco_entrada - (df.loc[i, "close"] + spread)) * num_acoes  # Cálculo do acumulado para Short
                df.loc[i, "Estado"] = "Vendido"  # Marca a operação como "Vendido"
            df.loc[i, "Acumulado"] = acumulado  # Atualiza o valor acumulado

    # Finaliza o backtest e exibe uma mensagem
    print("Simulação do backtest concluída.\n")


    # =============================================================================
    # 19. Análise dos Resultados dos Últimos 30 Dias
    # =============================================================================
    # Exibe a mensagem indicando que a análise dos resultados dos últimos 30 dias está começando
    print("Analisando resultados dos últimos 30 dias...")

    # Adiciona uma nova coluna 'date' no DataFrame, extraindo apenas a data (sem o horário) da coluna 'time'
    df['date'] = df['time'].dt.date

    # Agrupa os resultados diários, somando o lucro/prejuízo, contando o número de transações,
    # o número de transações com lucro e o número de transações com prejuízo
    daily_results = df.groupby('date').agg(
        total_profit_loss=('Lucro/Prejuízo', 'sum'),  # Soma do lucro/prejuízo diário
        num_transactions=('Lucro/Prejuízo', 'size'),  # Conta o número total de transações por dia
        num_profit_transactions=('Lucro/Prejuízo', lambda x: (x > 0).sum()),  # Conta transações com lucro
        num_loss_transactions=('Lucro/Prejuízo', lambda x: (x < 0).sum())  # Conta transações com prejuízo
    )

    # Encontra a data mais recente nos resultados diários
    max_date = daily_results.index.max()

    # Filtra os resultados dos últimos 30 dias, baseado na data máxima encontrada
    last_30_days = daily_results[daily_results.index >= (max_date - pd.Timedelta(days=30))]

    # Reseta o índice para obter um DataFrame com um índice numérico sequencial
    last_30_days_df = last_30_days.reset_index()

    # Exibe os resultados dos últimos 30 dias
    print("\n🔍 Análise dos Resultados dos Últimos 30 Dias:")
    print(last_30_days_df.to_string(index=False))

    # Calcula o total de lucro/prejuízo nos últimos 30 dias
    total_30 = last_30_days['total_profit_loss'].sum()

    # Calcula a média diária de lucro/prejuízo nos últimos 30 dias
    mean_30 = last_30_days['total_profit_loss'].mean()

    # Calcula o desvio padrão dos resultados diários
    std_30 = daily_results['total_profit_loss'].std()

    # Exibe o total, média diária e desvio padrão dos resultados dos últimos 30 dias
    print(f"\nTotal (30 dias): R$ {total_30:.2f}")
    print(f"Média Diária: R$ {mean_30:.2f}")
    print(f"Desvio Padrão: R$ {std_30:.2f}\n")


    # =============================================================================
    # 20. Cálculo da Média Mensal de Ganho
    # =============================================================================
    # Exibe a mensagem indicando que o cálculo da média mensal de ganho está começando
    print("Calculando média mensal de ganho...")

    # Cria uma nova coluna 'month' no DataFrame, convertendo a coluna 'time' para o período mensal
    # Isso ajuda a agrupar os dados por mês, ignorando o dia e o ano
    df['month'] = df['time'].dt.to_period('M')

    # Agrupa os dados por mês, calculando o lucro/prejuízo total, o número de transações, e o número de transações lucrativas e prejudiciais
    monthly_results = df.groupby('month').agg(
        total_profit_loss=('Lucro/Prejuízo', 'sum'),  # Soma do lucro/prejuízo de cada mês
        num_transactions=('Lucro/Prejuízo', 'size'),  # Conta o número total de transações de cada mês
        num_profit_transactions=('Lucro/Prejuízo', lambda x: (x > 0).sum()),  # Conta transações com lucro
        num_loss_transactions=('Lucro/Prejuízo', lambda x: (x < 0).sum())  # Conta transações com prejuízo
    )

    # Calcula a média do lucro/prejuízo total mensal
    monthly_mean = monthly_results['total_profit_loss'].mean()

    # Exibe os resultados mensais no console
    print("\n🔍 Média Mensal de Ganho:")
    # Exibe o DataFrame de resultados mensais, removendo o índice
    print(monthly_results.reset_index().to_string(index=False))

    # Exibe a média mensal de lucro/prejuízo
    print(f"\nMédia Mensal: R$ {monthly_mean:.2f}\n")


    # =============================================================================
    # 21. Estatísticas das Operações Fechadas
    # =============================================================================
    # Exibe a mensagem indicando que o cálculo das estatísticas das operações fechadas está começando
    print("Calculando estatísticas das operações fechadas...")

    # Filtra o DataFrame para incluir apenas as linhas onde a coluna 'Estado' indica que a operação foi fechada
    # Os estados "Fechado Horário", "Reversão Long" e "Reversão Short" indicam operações finalizadas
    fechadas = df[df["Estado"].isin(["Fechado Horário", "Reversão Long", "Reversão Short"])]

    # Calcula a quantidade de operações com lucro, contando quantos valores na coluna 'Lucro/Prejuízo' são positivos
    qtd_profit = (fechadas["Lucro/Prejuízo"] > 0).sum()

    # Calcula a quantidade de operações com prejuízo, contando quantos valores na coluna 'Lucro/Prejuízo' são negativos
    qtd_loss = (fechadas["Lucro/Prejuízo"] < 0).sum()

    # Calcula o somatório dos lucros, somando os valores da coluna 'Lucro/Prejuízo' que são positivos
    soma_profit = fechadas.loc[fechadas["Lucro/Prejuízo"] > 0, "Lucro/Prejuízo"].sum()

    # Calcula o somatório dos prejuízos, somando os valores da coluna 'Lucro/Prejuízo' que são negativos
    soma_loss = fechadas.loc[fechadas["Lucro/Prejuízo"] < 0, "Lucro/Prejuízo"].sum()

    # Calcula o maior lucro obtido em uma única operação
    max_gain = fechadas["Lucro/Prejuízo"].max()

    # Calcula o maior prejuízo obtido em uma única operação
    max_loss = fechadas["Lucro/Prejuízo"].min()

    # Exibe as estatísticas calculadas no console
    print("\n🔍 Estatísticas das Operações Fechadas:")
    # Exibe a quantidade de operações com lucro
    print(f"Quantidade de operações com lucro: {qtd_profit}")
    # Exibe a quantidade de operações com prejuízo
    print(f"Quantidade de operações com prejuízo: {qtd_loss}")
    # Exibe o somatório dos lucros
    print(f"Somatório dos lucros: R$ {soma_profit:.2f}")
    # Exibe o somatório dos prejuízos
    print(f"Somatório dos prejuízos: R$ {soma_loss:.2f}")
    # Exibe o ganho máximo em uma única operação
    print(f"Ganho máximo em uma operação: R$ {max_gain:.2f}")
    # Exibe o prejuízo máximo em uma única operação
    print(f"Prejuízo máximo em uma operação: R$ {max_loss:.2f}\n")


    # =============================================================================
    # 22. Exibir Resumo Geral dos Resultados e Salvar CSV
    # =============================================================================
    # Inicia o cálculo do resumo geral dos resultados.
    print("Calculando resumo geral dos resultados...")

    # Calcula o total de lucro ou prejuízo gerado por todas as operações, somando os valores na coluna 'Lucro/Prejuízo'
    total_lucro = df["Lucro/Prejuízo"].sum()

    # Calcula o ROI (Retorno sobre Investimento), usando o total de lucro ou prejuízo e o capital inicial.
    # O ROI é calculado como o lucro total dividido pelo capital inicial multiplicado por 100 para obter o valor percentual.
    roi = (total_lucro / capital_inicial) * 100

    # Calcula o capital final após o lucro ou prejuízo, somando o capital inicial com o total de lucro ou prejuízo.
    capital_final = capital_inicial + total_lucro

    # Exibe o valor do capital inicial, com formatação para mostrar o valor em reais (R$).
    print(f"\n💰 Capital Inicial: R$ {capital_inicial:,.2f}")

    # Exibe o valor total de lucro ou prejuízo gerado pelas operações.
    print(f"📈 Lucro/Prejuízo Total: R$ {total_lucro:.2f}")

    # Exibe o ROI calculado, indicando o retorno percentual do investimento.
    print(f"📊 ROI: {roi:.2f}%")

    # Exibe o capital final, ou seja, o capital inicial somado ao lucro ou prejuízo obtido.
    print(f"🔹 Capital Final: R$ {capital_final:.2f}\n")


# =============================================================================
# Loop principal
# =============================================================================
# Percorre todos os símbolos na lista 'symbols' e processa cada um deles
for symbol in symbols:
    try:
        # Inicia o processamento para o símbolo atual, imprimindo uma mensagem
        print(f"Processamento inciado: {symbol} !")

        # Chama a função 'processar_acao' para realizar o processamento dos dados
        # para o símbolo atual (a função 'processar_acao' é responsável por diversas operações no símbolo)
        processar_acao(symbol)

        # Após a execução da função 'processar_acao', indica que o processamento foi concluído com sucesso
        print(f"Processamento Concluído: {symbol} !")

    except Exception as e:
        # Se ocorrer algum erro ao processar o símbolo, captura a exceção e imprime uma mensagem de erro
        print(f"Erro ao processar {symbol}: {e}")

# =============================================================================
# Salvar Excel consolidado
# =============================================================================
# Define o nome do arquivo Excel onde os resultados consolidados serão salvos
excel_consolidado = "Resultados_Modelos.xlsx"

# Cria um DataFrame 'df_resultados' a partir da lista 'resultados_finais', que contém os resultados de cada símbolo
df_resultados = pd.DataFrame(resultados_finais)

# Utiliza o ExcelWriter com o engine 'xlsxwriter' para salvar o DataFrame em um arquivo Excel
with pd.ExcelWriter(excel_consolidado, engine='xlsxwriter') as writer:
    # Salva os dados do DataFrame na planilha "Resumo_Modelos" do arquivo Excel
    # O parâmetro 'index=False' indica que o índice não será incluído no arquivo Excel
    df_resultados.to_excel(writer, index=False, sheet_name="Resumo_Modelos")

# Após o processo de salvar o Excel, imprime uma mensagem indicando o caminho do arquivo salvo
print(f"\n✅ Resultados consolidados salvos em: {excel_consolidado}")

