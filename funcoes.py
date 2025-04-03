import pandas as pd
import matplotlib.pyplot as plt


def plot_importance(importances, features, title, filename):
    """
    Gera e salva um gráfico de barras para a importância das features com base nos valores fornecidos.

    Parâmetros:
        importances: Lista ou array contendo os valores de importância das features.
        features: Lista de strings contendo os nomes das features.
        title: Título do gráfico a ser gerado.
        filename: Caminho e nome do arquivo onde o gráfico será salvo.

    Passos:
        1. Cria uma série do pandas para mapear a importância das features.
        2. Ordena as features por sua importância de forma decrescente.
        3. Plota um gráfico de barras da importância das features.
        4. Salva o gráfico gerado no arquivo especificado.
    """

    # Cria uma série com as importâncias das features, associando cada valor de importância a sua respectiva feature.
    importance_series = pd.Series(importances, index=features).sort_values(ascending=False)

    # Plota o gráfico de barras com as importâncias das features.
    importance_series.plot(kind='bar', figsize=(10, 6), title=title)

    # Salva o gráfico gerado no arquivo especificado.
    plt.savefig(filename)

    # Fecha a figura do gráfico para liberar recursos.
    plt.close()


def testar_thresholds_importancia(modelo_base, X_df, y, symbol_timeframe_path, symbol, timeframe,
                                      thresholds=[-0.0005, 0.0, 0.0005, 0.001, 0.002]):
        """
        Função para testar diferentes thresholds de importância de features e analisar o impacto na acurácia do modelo.

        Parâmetros:
        - modelo_base: modelo RandomForest treinado para calcular a importância das features.
        - X_df: DataFrame contendo as features.
        - y: Variável alvo (target).
        - symbol_timeframe_path: Caminho onde os gráficos serão salvos.
        - symbol: Símbolo do ativo (ex: 'AAPL').
        - timeframe: Timeframe usado.
        - thresholds: Lista de thresholds para testar (default: [-0.0005, 0.0, 0.0005, 0.001, 0.002]).

        Retorna:
        - df_resultados: DataFrame com os resultados de acurácia para cada threshold e número de features selecionadas.
        """
        # Lista para armazenar os resultados
        resultados = []

        # Calcula a importância das features usando a técnica de Permutation Importance
        perm_result = permutation_importance(modelo_base, X_df.values, y, n_repeats=10, random_state=42)
        importancias = pd.Series(perm_result.importances_mean, index=X_df.columns)

        # Loop para testar diferentes thresholds
        for threshold in thresholds:
            # Seleciona features com importância maior que o threshold
            selected_features = importancias[importancias > threshold].index.tolist()

            # Prepara os dados selecionando apenas as features acima do threshold
            X_sel = X_df[selected_features]

            # Imputa valores ausentes usando a média e normaliza os dados
            imputer = SimpleImputer(strategy='mean')
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(imputer.fit_transform(X_sel))

            # Divide os dados em treino e teste
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.3, stratify=y, random_state=42)

            # Cria e treina o modelo RandomForest
            model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            model.fit(X_train, y_train)

            # Avalia o modelo no conjunto de teste e calcula a acurácia
            acc = accuracy_score(y_test, model.predict(X_test))

            # Armazena o threshold, número de features selecionadas e a acurácia
            resultados.append((threshold, len(selected_features), acc))
