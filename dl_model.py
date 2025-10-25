import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

processed_df = pd.read_csv("/kaggle/working/lc_processed_sample.csv")

X = processed_df.drop(columns=['target']).values.astype('float32')
y = processed_df['target'].values.astype('int')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.AUC(name='AUC')]
)

es = EarlyStopping(monitor='val_AUC', patience=3, mode='max', restore_best_weights=True)

cw = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
cw = dict(enumerate(cw))

history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=20,
    batch_size=512,
    callbacks=[es],
    class_weight=cw,
    verbose=1
)

y_pred = model.predict(X_test)
y_pred_labels = (y_pred > 0.5).astype(int)

auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred_labels)
print(f"\nTest AUC: {auc:.4f}")
print(f"Test F1-Score: {f1:.4f}")

cm = confusion_matrix(y_test, y_pred_labels)
ConfusionMatrixDisplay(cm, display_labels=["Fully Paid", "Defaulted"]).plot(cmap='Blues')
plt.title("Confusion Matrix")
plt.show()

plt.figure(figsize=(10,4))
plt.plot(history.history['AUC'], label='Train AUC')
plt.plot(history.history['val_AUC'], label='Val AUC')
plt.xlabel('Epochs')
plt.ylabel('AUC')
plt.legend()
plt.title("AUC Learning Curve")
plt.show()

model.save("/kaggle/working/dl_model.h5")
print("Saved trained DL model to dl_model.h5")
