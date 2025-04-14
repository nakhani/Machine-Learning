import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("dataset/Dataset.csv")

df['S0'] = df['S0'] / 400
df['A0'] = df['A0'] / 400
df['S1'] = df['S1'] /400
df['A1'] = df['A1'] / 400

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

scaler = StandardScaler()
X_ = scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.2, random_state=42)
x_test, x_validation, y_test, y_validation = train_test_split(x_test, y_test, test_size=.25, random_state=42)


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, input_dim=x_train.shape[1], activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(64, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(32, activation=tf.keras.activations.relu),
    tf.keras.layers.Dense(4, activation=tf.keras.activations.softmax)
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

train_result = model.fit(x_train, y_train, epochs=200, validation_data=(x_validation, y_validation),batch_size=32, callbacks=[early_stopping])
test_result = model.evaluate(x_test, y_test)


fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#2b2b2b')  

ax.plot(train_result.history['loss'], label='Train Loss', color='cyan', linewidth=2)
ax.plot(train_result.history['val_loss'], label='Validation Loss', color='magenta', linewidth=2)

ax.set_title('Training and Validation Loss', color='#e0e0e0')
ax.set_xlabel('Epochs', color='#e0e0e0')
ax.set_ylabel('Loss', color='#e0e0e0')
ax.tick_params(colors='#e0e0e0')
ax.grid(color='#555555', linestyle='--', linewidth=0.5, alpha=0.6)
ax.set_facecolor('#2b2b2b') 
ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='#e0e0e0')

plt.tight_layout()
plt.savefig("output1.png")


fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#2b2b2b')  

ax.plot(train_result.history['accuracy'], label='Train Accuracy', color='yellow', linewidth=2)
ax.plot(train_result.history['val_accuracy'], label='Validation Accuracy', color='lime', linewidth=2)

ax.set_title('Training and Validation Accuracy', color='#e0e0e0')
ax.set_xlabel('Epochs', color='#e0e0e0')
ax.set_ylabel('Accuracy', color='#e0e0e0')
ax.tick_params(colors='#e0e0e0')
ax.grid(color='#555555', linestyle='--', linewidth=0.5, alpha=0.6)
ax.set_facecolor('#2b2b2b') 
ax.legend(facecolor='#2b2b2b', edgecolor='white', labelcolor='#e0e0e0')

plt.tight_layout()
plt.savefig("output2.png")


print(f'Test results: {test_result}')

model.save('model.keras')
model.save('model.h5')
