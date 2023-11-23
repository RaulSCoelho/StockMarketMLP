import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import time

# Start the timer
start_time = time.time()

# Leitura do arquivo CSV
data = pd.read_csv('acoes.csv', encoding='ISO-8859-1', sep=';', decimal=',')
# Convertendo colunas de data
data['ref.date'] = pd.to_datetime(data['ref.date'])
# Tratando valores NA
# Preenche com o valor anterior (forward fill)
data = data.ffill()
# Removendo linhas duplicadas
data = data.drop_duplicates()

numAmostras, numCols = data.shape

# Selecionando colunas para usar como features e target
features = ['price.open', 'price.high', 'price.low', 'volume']
target = ['price.close']

X = data[features]
y = data[target]

# Normalização manual dos dados
X_norm = (X - X.min()) / (X.max() - X.min())
y_norm = (y - y.min()) / (y.max() - y.min())

# Dividindo os dados em treino e teste
train_size = int(0.8 * numAmostras)

X_train = X_norm[:train_size]
y_train = y_norm[:train_size]
X_test = X_norm[train_size:]
y_test = y_norm[train_size:]

# Definição do número de neurônios
m = len(features)  # Número de neurônios na camada de entrada
N = int(m * 2)  # Número de neurônios na camada escondida
L = 1  # Número de neurônios na camada de saída

# Inicialização dos pesos
W1 = np.random.random((N, m + 1))  # Camada de entrada para a camada escondida
W2 = np.random.random((N, N + 1))  # Camada de entrada para a camada escondida
W3 = np.random.random((L, N + 1))  # Camada escondida para a camada de saída

numEpocas = 150  # Número de épocas
eta = 0.02  # Taxa de aprendizado
q = train_size  # Número de padrões de treinamento

# Array para armazenar os erros
E = np.zeros(q)
Etm = np.zeros(numEpocas)  # Erro total médio
bias = 1  # Bias

# Treinamento da Rede
for i in range(numEpocas):
    for j in range(q):
        # Preparando a entrada com bias
        Xb = np.hstack((bias, X_train.iloc[j].values))

        # Saída da primeira Camada Escondida
        o1 = np.tanh(W1.dot(Xb))
        o1b = np.insert(o1, 0, bias)  # Incluindo o bias

        # Saída da segunda Camada Escondida
        o2 = np.tanh(W2.dot(o1b))
        o2b = np.insert(o2, 0, bias)  # Incluindo o bias

        # Saída da Rede Neural (Camada de Saída)
        Y = np.tanh(W3.dot(o2b))

        # Calculando o erro
        e = y_train.iloc[j] - Y
        E[j] = (e.transpose().dot(e)) / 2

        # Backpropagation
        # Erro na camada de saída
        delta3 = np.diag(e).dot((1 - Y * Y))

        # Erro propagado para a segunda camada escondida
        vdelta3 = (W3.transpose()).dot(delta3)
        delta2 = np.diag(1 - o2b * o2b).dot(vdelta3)

        # Erro propagado para a primeira camada escondida
        vdelta2 = (W2.transpose()).dot(delta2[1:])
        delta1 = np.diag(1 - o1b * o1b).dot(vdelta2)

        # Atualização dos pesos
        W1 = W1 + eta * (np.outer(delta1[1:], Xb))
        W2 = W2 + eta * (np.outer(delta2[1:], o1b))
        W3 = W3 + eta * (np.outer(delta3, o2b))

    # Calculo da média dos erros
    Etm[i] = E.mean()

# Teste da Rede
Error_Test = np.zeros(len(y_test))
Predictions = np.zeros(len(y_test))  # Armazenará as previsões da rede

for i in range(len(y_test)):
    # Preparando a entrada com bias
    Xb = np.hstack((bias, X_test.iloc[i].values))

    # Saída da primeira Camada Escondida
    o1 = np.tanh(W1.dot(Xb))
    o1b = np.insert(o1, 0, bias)  # Incluindo o bias

    # Saída da segunda Camada Escondida
    o2 = np.tanh(W2.dot(o1b))
    o2b = np.insert(o2, 0, bias)  # Incluindo o bias

    # Saída da Rede Neural (Camada de Saída)
    Y = np.tanh(W3.dot(o2b))

    # Armazenando as previsões e calculando os erros
    Predictions[i] = Y
    Error_Test[i] = y_test.iloc[i] - Y

# End the timer
end_time = time.time()

plt.figure(figsize=(10, 5))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Valores Reais')
plt.scatter(range(len(y_test)), Predictions,
            color='red', label='Previsões', alpha=0.5)
plt.title('Previsões vs Valores Reais')
plt.xlabel('Índice')
plt.ylabel('Preço de Fechamento')
plt.legend()
plt.show()

plt.xlabel("Épocas")
plt.ylabel("Erro Médio")
plt.plot(Etm, color='b')
plt.show()

print()
print("Erros de Teste:", Error_Test)
print()

# Cálculo da porcentagem de erro para cada item
y_test_flatten = y_test.values.flatten()
errors_percent = np.abs(Predictions - y_test_flatten) / y_test_flatten * 100
print("Erros Percentual:", errors_percent)
print()

errors_percent_rounded = np.round(errors_percent, 2)
# Cálculo da média das porcentagens de erro
media_erros = np.mean(errors_percent_rounded)
# Cálculo da mediana das porcentagens de erro
mediana_erros = np.median(errors_percent_rounded)
# Cálculo da moda das porcentagens de erro
moda_erros = stats.mode(errors_percent_rounded)

print(f"Erro médio: {media_erros:.2f}%")
print(f"Erro mediano: {mediana_erros:.2f}%")
print(
    f"Erro modal: {moda_erros.mode:.2f}% (ocorrências: {moda_erros.count})")
print()
print(f"Erro maior: {errors_percent_rounded.max():.2f}%")
print(f"Erro menor: {errors_percent_rounded.min():.2f}%")

duration = end_time - start_time
hours = int(duration // 3600)
minutes = int((duration % 3600) // 60)
seconds = int(duration % 60)

print(f"Tempo de execução: {hours:02d}:{minutes:02d}:{seconds:02d}")
