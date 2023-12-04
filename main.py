import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import seaborn as sns
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier
from art.defences.trainer import AdversarialTrainer
from art import config

tf.compat.v1.disable_eager_execution()
config.USE_TF_KERAS = True  # Используем tensorflow'ский keras

# Загрузка и предобработка CIFAR-10 данных
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

class_names = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# Модель
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# Оценка модели на исходных данных
y_scores = model.predict(x_test)
y_pred = np.argmax(y_scores, axis=1)
y_true = y_test.reshape(-1)

# Матрица ошибок
conf_matrix = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
plt.yticks(np.arange(len(class_names)), class_names, rotation=0)
plt.title("Confusion Matrix (Original)")
plt.show()

# ROC-кривые
y_true_binary = label_binarize(y_true, classes=range(10))
fpr = {}
tpr = {}
roc_auc = {}
for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_true_binary[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.plot(fpr[i], tpr[i], label=f'Class {class_names[i]} (AUC = {roc_auc[i]:.8f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title("ROC Curves (Original)")
plt.show()

# Точность
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy (Original): {accuracy:.8f}")

# Атака Fast Gradient Method (FGSM)
estimator = KerasClassifier(model=model)
attack = FastGradientMethod(estimator, eps=0.05)
x_test_adv = attack.generate(x_test)

# Оценка модели на искаженных данных
y_scores_adv = model.predict(x_test_adv)
y_pred_adv = np.argmax(y_scores_adv, axis=1)

# Матрица ошибок
conf_matrix_adv = confusion_matrix(y_true, y_pred_adv)

plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix_adv, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
plt.yticks(np.arange(len(class_names)), class_names, rotation=0)
plt.title("Confusion Matrix (Adversarial)")
plt.show()

# ROC-кривые
y_true_binary_adv = label_binarize(y_true, classes=range(10))
fpr_adv = {}
tpr_adv = {}
roc_auc_adv = {}
for i in range(10):
    fpr_adv[i], tpr_adv[i], _ = roc_curve(y_true_binary_adv[:, i], y_scores_adv[:, i])
    roc_auc_adv[i] = auc(fpr_adv[i], tpr_adv[i])

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.plot(fpr_adv[i], tpr_adv[i], label=f'Class {class_names[i]} (AUC = {roc_auc_adv[i]:.8f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title("ROC Curves (Adversarial)")
plt.show()

# Точность
accuracy_adv = accuracy_score(y_true, y_pred_adv)
print(f"Accuracy (Adversarial): {accuracy_adv:.8f}")

# Защищаем модель с помощью состязательного обучения с библиотеки ART
protected_model = AdversarialTrainer(estimator, attacks=attack)
protected_model.fit(x_train, y_train, nb_epochs=10)

# Оцениваем модель на защищенных данных
y_scores_protected = protected_model.predict(x_test)
y_pred_protected = np.argmax(y_scores_protected, axis=1)

# Матрица ошибок
conf_matrix_protected = confusion_matrix(y_true, y_pred_protected)

plt.figure(figsize=(10, 10))
sns.heatmap(conf_matrix_protected, annot=True, fmt="d", cmap="Blues", linewidths=0.5)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(np.arange(len(class_names)), class_names, rotation=90)
plt.yticks(np.arange(len(class_names)), class_names, rotation=0)
plt.title("Confusion Matrix (Protected)")
plt.show()

# ROC-кривые
y_true_binary_protected = label_binarize(y_true, classes=range(10))
fpr_protected = {}
tpr_protected = {}
roc_auc_protected = {}
for i in range(10):
    fpr_protected[i], tpr_protected[i], _ = roc_curve(y_true_binary_protected[:, i], y_scores_protected[:, i])
    roc_auc_protected[i] = auc(fpr_protected[i], tpr_protected[i])

plt.figure(figsize=(10, 10))
for i in range(10):
    plt.plot(fpr_protected[i], tpr_protected[i], label=f'Class {class_names[i]} (AUC = {roc_auc_protected[i]:.8f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.title("ROC Curves (Protected)")
plt.show()

# Точность
accuracy_protected = accuracy_score(y_true, y_pred_protected)
print(f"Accuracy (Protected): {accuracy_protected:.8f}")
