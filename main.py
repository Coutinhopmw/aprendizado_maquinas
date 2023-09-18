# Importar bibliotecas necessárias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# Carregar o dataset
dataset = pd.read_csv('card_transdata.csv')

# Dividir o dataset em recursos (X) e rótulos (y)
X = dataset[['distance_from_home', 'used_pin_number','distance_from_last_transaction']]
#X = dataset.drop('fraud', axis=1)
y = dataset['fraud']

# Dividir o dataset em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar os recursos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar e avaliar o modelo Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_predictions = nb_classifier.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

print("Naive Bayes:")
print("Acurácia:", nb_accuracy)
print("Relatório de Classificação:")
print(classification_report(y_test, nb_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_predictions))

# Treinar e avaliar o modelo Árvore de Decisão
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nÁrvore de Decisão:")
print("Acurácia:", dt_accuracy)
print("Relatório de Classificação:")
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_predictions))
print(classification_report(y_test, dt_predictions))

# Treinar e avaliar o modelo K-Nearest Neighbors (KNN)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("\nK-Nearest Neighbors (KNN):")
print("Acurácia:", knn_accuracy)
print("Relatório de Classificação:")
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions))

#Função para plotar as fronteiras de decisão
def plot_decision_boundary(X, y, classifier, title):
    h = .6  # Tamanho do passo na malha
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    Z = classifier.predict(np.c[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, marker='o', edgecolors='k')
    plt.title(title)

#Plotando as fronteiras de decisão usando as duas primeiras features
X_train_plot = X_train[:, :2]
X_test_plot = X_test[:, :2]

plt.figure(figsize=(15, 5))

#Fronteira de decisão do Naive Bayes
plt.subplot(131)
plot_decision_boundary(X_train_plot, y_train, nb_classifier, 'Fronteira de Decisão - Naive Bayes')

#Fronteira de decisão da Árvore de Decisão
plt.subplot(132)
plot_decision_boundary(X_train_plot, y_train, dt_classifier, 'Fronteira de Decisão - Árvore de Decisão')

#Fronteira de decisão do KNN
plt.subplot(133)
plot_decision_boundary(X_train_plot, y_train, knn_classifier, 'Fronteira de Decisão - KNN')

plt.tight_layout()
plt.show()
