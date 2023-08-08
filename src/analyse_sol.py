from CapsNet import CapsNet
from utils.load_data import load_mnist
import json
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

(X_train, y_train), (X_test , y_test) = load_mnist()
n_rounds = 10

directory = 'res'
if not os.path.exists(directory):
    os.makedirs(directory)

# SOLUTION CAPSNET
with open('../saved_ga/mnist-gpu0-hpc-2/solutions.json') as file:
    solutions = json.load(file)
    solution_capsnet = CapsNet(**solutions['Solution 2']['params'])
    solution_capsnet.summary()
    solution_capsnet.fit(X_train, y_train, tf.keras.optimizers.legacy.Adam(), train_metrics=['accuracy'], validation=(X_test, y_test))
solution_capsnet.save()
sol_inf_time = []
for i in range(n_rounds):
    start = time.time()
    solution_capsnet.predict(X_test)
    end = time.time()
    sol_inf_time.append((end - start) / len(X_test) * 1000)
trainable_params = sum([tf.keras.backend.count_params(w) for w in solution_capsnet.trainable_weights])


# BASELINE CAPSNET
params = {
    "epochs": 10,
    "epsilon": 1e-7,
    "m_minus": 0.1,
    "m_plus": 0.9,
    "lambda_": 0.5,
    "alpha": 0.0005,
    "no_of_conv_kernels": 128,
    "no_of_primary_capsules": 32,
    "no_of_secondary_capsules": 10,
    "primary_capsule_vector": 8,
    "secondary_capsule_vector": 16,
    "r":3,
    "dense_1": 512,
    "dense_2": 1024
}
baseline_capsnet = CapsNet(**params)
baseline_capsnet._name = 'baseline_capsnet'
baseline_capsnet.compile()
baseline_capsnet.summary()
baseline_capsnet.fit(X_train, y_train, tf.keras.optimizers.legacy.Adam(), train_metrics=['accuracy'], validation=(X_test, y_test))
baseline_capsnet.save()
caps_inf_time = []
for i in range(n_rounds):
    start = time.time()
    baseline_capsnet.predict(X_test)
    end = time.time()
    caps_inf_time.append((end - start) / len(X_test) * 1000)


# BASELINE CNN
basline_cnn = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
basline_cnn.summary()
basline_cnn.compile(tf.optimizers.legacy.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
tf_y_train = tf.keras.utils.to_categorical(y_train)
tf_y_test = tf.keras.utils.to_categorical(y_test)
history = basline_cnn.fit(X_train, tf_y_train, epochs=10, batch_size=32, validation_data=(X_test, tf_y_test))
basline_cnn.save("../saved_models/baseline_cnn")
df_baseline_cnn = pd.DataFrame(history.history)
plot_baseline = df_baseline_cnn.plot(title="CNN baseline model")
plot_baseline.set_ylim(0, 1)
plot_baseline.set_xlim(0, 9)
plt.savefig('../saved_models/baseline_cnn/assets/training_metrics')
cnn_inf_time = []
for i in range(n_rounds):
    start = time.time()
    basline_cnn.predict(X_test)
    end = time.time()
    cnn_inf_time.append((end - start) / len(X_test) * 1000)

# PLOT TRAINING METRICS
solution_capsnet.training_metrics = {
    'loss': solution_capsnet.training_metrics['loss'],
    'accuracy': solution_capsnet.training_metrics['accuracy'],
    'val_loss': solution_capsnet.training_metrics['val_loss'],
    'val_accuracy': solution_capsnet.training_metrics['val_accuracy'],
}
baseline_capsnet.training_metrics = {
    'loss': baseline_capsnet.training_metrics['loss'],
    'accuracy': baseline_capsnet.training_metrics['accuracy'],
    'val_loss': baseline_capsnet.training_metrics['val_loss'],
    'val_accuracy': baseline_capsnet.training_metrics['val_accuracy'],
}
fig, axes = plt.subplots(1, 3, figsize=(12, 5))
df_baseline = pd.DataFrame(history.history)
plot_baseline = df_baseline.plot(ax=axes[0], title="CNN model", legend=False)
axes[0].set_ylim(0, 1)
axes[0].set_xlim(0, 9)
df_capsnet = pd.DataFrame(baseline_capsnet.training_metrics)
plot_capsnet = df_capsnet.plot(ax=axes[1], title="CapsNet baseline model", legend=False)
axes[1].set_ylim(0, 1)
axes[1].set_xlim(0, 6)
df_capsnet_nas = pd.DataFrame(solution_capsnet.training_metrics)
plot_capsnet_nas = df_capsnet_nas.plot(ax=axes[2], title="CapsNet solution model", legend=False)
axes[2].set_ylim(0, 1)
axes[2].set_xlim(0, 5)
axes[0].set_xlabel("Epochs")
axes[1].set_xlabel("Epochs")
axes[2].set_xlabel("Epochs")
handles, labels = axes[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.1), fancybox=True, ncol=4)
plt.tight_layout()
plt.savefig('./res/training_metrics', bbox_inches='tight')

# PLOT INFERENCE TIME
df = pd.DataFrame({
    "CNN": cnn_inf_time,
    "CapsNet baseline": caps_inf_time,
    "CapsNet solution": sol_inf_time
})
plt.figure(figsize=(8, 6))
plt.boxplot(df.values, labels=df.columns)
plt.title('Inference Time Comparison')
plt.xlabel('Model')
plt.ylabel('Inference Time (ms)')
plt.grid(True)
plt.savefig('./res/inference_time')

# PLOT ACCURACY
df = pd.DataFrame({
    'CNN': {'acc': history.history['accuracy'][-1], 'val_acc': history.history['val_accuracy'][-1]},
    'Caps': {'acc': baseline_capsnet.training_metrics['accuracy'][-1], 'val_acc': baseline_capsnet.training_metrics['val_accuracy'][-1]},
    'Caps solution': {'acc': solution_capsnet.training_metrics['accuracy'][-1], 'val_acc': solution_capsnet.training_metrics['val_accuracy'][-1]},
})
df = df.T
df.plot(kind='bar')
plt.xlabel('Model')
plt.ylabel('Accuracy')
plt.title('Accuracy Comparison')
plt.legend(title='Metric', loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.gcf().set_size_inches(8, 5)
plt.savefig('./res/accuracy_comparison')
