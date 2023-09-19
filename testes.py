# Importar bibliotecas necessárias
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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
print("Acuracia:", nb_accuracy)
print("Relatorio de Classificaçao:")
print(classification_report(y_test, nb_predictions))
print("Confusion Matrix:\n", confusion_matrix(y_test, nb_predictions))

# Treinar e avaliar o modelo Árvore de Decisão
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
dt_predictions = dt_classifier.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

print("\nArvore de Decisao:")
print("Acuracia:", dt_accuracy)
print("Relatorio de Classificaçao:")
print("Confusion Matrix:\n", confusion_matrix(y_test, dt_predictions))
print(classification_report(y_test, dt_predictions))

# Treinar e avaliar o modelo K-Nearest Neighbors (KNN)
knn_classifier = KNeighborsClassifier(n_neighbors=5)
knn_classifier.fit(X_train, y_train)
knn_predictions = knn_classifier.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

print("\nK-Nearest Neighbors (KNN):")
print("Acuracia:", knn_accuracy)
print("Relatorio de Classificação:")
print("Confusion Matrix:\n", confusion_matrix(y_test, knn_predictions))
print(classification_report(y_test, knn_predictions))

def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_test, nb_predictions, labels=['Não Fraude', 'Fraude'])
