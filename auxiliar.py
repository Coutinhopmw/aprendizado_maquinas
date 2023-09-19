
def plot_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

plot_confusion_matrix(y_test, nb_predictions, labels=['Não Fraude', 'Fraude'])



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
