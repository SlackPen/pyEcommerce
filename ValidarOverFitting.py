import numpy as np
from sklearn.metrics import f1_score
from sklearn.utils import resample
from scipy import stats

# Suponha que já temos:
# y_train_true, y_train_pred
# y_test_true, y_test_pred

# --- 1. Calculando F1 médio ---
f1_train = f1_score(y_train_true, y_train_pred, average='macro')
f1_test = f1_score(y_test_true, y_test_pred, average='macro')

print(f"F1 Treino: {f1_train:.3f}")
print(f"F1 Teste:  {f1_test:.3f}")

# --- 2. Bootstrapping para intervalo de confiança do F1 ---
def bootstrap_ci(y_true, y_pred, metric_fn, n_iterations=1000, alpha=0.05):
    scores = []
    n = len(y_true)
    for _ in range(n_iterations):
        indices = resample(range(n), replace=True, n_samples=n)
        score = metric_fn(np.array(y_true)[indices], np.array(y_pred)[indices])
        scores.append(score)
    lower = np.percentile(scores, 100 * alpha/2)
    upper = np.percentile(scores, 100 * (1 - alpha/2))
    return np.mean(scores), (lower, upper)

f1_test_mean, f1_test_ci = bootstrap_ci(y_test_true, y_test_pred, 
                                        lambda y1, y2: f1_score(y1, y2, average='macro'))

print(f"F1 Teste (Bootstrap): {f1_test_mean:.3f} ± ({f1_test_ci[0]:.3f}, {f1_test_ci[1]:.3f})")

# --- 3. Teste t pareado (assumindo que temos métricas por amostra) ---
# ex: probabilidade ou F1 local por amostra
sample_f1_train = [f1_score([yt], [yp], average='macro') for yt, yp in zip(y_train_true, y_train_pred)]
sample_f1_test  = [f1_score([yt], [yp], average='macro') for yt, yp in zip(y_test_true, y_test_pred)]

t_stat, p_value = stats.ttest_rel(sample_f1_train, sample_f1_test)
print(f"Teste t pareado: t = {t_stat:.3f}, p = {p_value:.4f}")
if p_value < 0.05:
    print("➡️ Diferença estatisticamente significativa: possível overfitting.")
else:
    print("✅ Sem evidência estatística forte de overfitting.")




# VISUALIZAÇÃO DAS CURVAS
import matplotlib.pyplot as plt

train_sizes = np.linspace(0.1, 1.0, 10)
train_scores, test_scores = [], []

for frac in train_sizes:
    idx = int(frac * len(y_train_true))
    f1_train = f1_score(y_train_true[:idx], y_train_pred[:idx], average='macro')
    f1_test = f1_score(y_test_true[:idx], y_test_pred[:idx], average='macro')
    train_scores.append(f1_train)
    test_scores.append(f1_test)

plt.plot(train_sizes, train_scores, label="Treino")
plt.plot(train_sizes, test_scores, label="Teste")
plt.xlabel("Tamanho relativo do treino")
plt.ylabel("F1 Score")
plt.title("Curva de Aprendizado")
plt.legend()
plt.grid()
plt.show()

