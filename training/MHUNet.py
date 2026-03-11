"""
MultiViewUNet Training Script

This script trains the MultiViewUNet model to predict hemodynamic
quantities (e.g., RRT, TAWSS, OSI, ECAP) from geometry-derived images.

Input  : Curvature images
Output : Hemodynamic maps (RRT in this experiment)

"""

# ============================================================
# System Information
# ============================================================

import subprocess
from psutil import virtual_memory

try:
    gpu_info = subprocess.check_output(['nvidia-smi'], encoding='utf-8')
    print(gpu_info)
except:
    print('GPU not detected or nvidia-smi unavailable.')

ram_gb = virtual_memory().total / 1e9
print(f"Available RAM: {ram_gb:.1f} GB\n")


# ============================================================
# Imports
# ============================================================

import os
import sys
import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from tqdm import tqdm

matplotlib.use('Agg')
plt.rcParams["font.size"] = 16

# Add path for networks.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from networks import MultiViewUNet

PROBLEM = "Curvature_2_RRT"
MODEL_NAME = "MHUNet"

# ============================================================
# Paths
# ============================================================

DATASET_DIR = os.path.join(BASE_DIR, "data", "images")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
PRED_DIR    = os.path.join(BASE_DIR, "predictions")
TRAIN_DIR = "Train"
TEST_DIR = "Test"

INPUT_DIR = "Curvature"
TARGET_DIR = "RRT"

IMG_SIZE = 768
BATCH_SIZE = 2
BUFFER_SIZE = 100
VAL_SPLIT = 0.2

LEARNING_RATE = 0.0001
N_EPOCHS = 300
PATIENCE = 30

EXP_NAME = f"{PROBLEM}_{MODEL_NAME}_I{IMG_SIZE}_B{BATCH_SIZE}_LR{LEARNING_RATE}"

print(f"Experiment: {EXP_NAME}")
print(f"Input Directory: {INPUT_DIR}")
print(f"Target Directory: {TARGET_DIR}")


# ============================================================
# Dataset Loader
# ============================================================

def load_data_from_dir(path: str):

    return tf.keras.utils.image_dataset_from_directory(
        directory=path,
        labels=None,
        color_mode='rgb',
        batch_size=BATCH_SIZE,
        image_size=(IMG_SIZE, IMG_SIZE),
        shuffle=False,
        seed=42
    )


trainX = load_data_from_dir(os.path.join(DATASET_PATH, TRAIN_DIR, INPUT_DIR))
trainY = load_data_from_dir(os.path.join(DATASET_PATH, TRAIN_DIR, TARGET_DIR))

testX = load_data_from_dir(os.path.join(DATASET_PATH, TEST_DIR, INPUT_DIR))
testY = load_data_from_dir(os.path.join(DATASET_PATH, TEST_DIR, TARGET_DIR))

train_ds = tf.data.Dataset.zip((trainX, trainY))
test_ds = tf.data.Dataset.zip((testX, testY))


# ============================================================
# Normalization
# ============================================================

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(
    lambda x, y: (normalization_layer(x), normalization_layer(y))
)

test_ds = test_ds.map(
    lambda x, y: (normalization_layer(x), normalization_layer(y))
)


# ============================================================
# Data Augmentation
# ============================================================

class Augment(tf.keras.layers.Layer):

    def __init__(self, seed=42):
        super().__init__()

        self.augment_inputs = tf.keras.layers.RandomZoom(
            (-0.1, -0.7), seed=seed
        )

        self.augment_labels = tf.keras.layers.RandomZoom(
            (-0.1, -0.7), seed=seed
        )

    def call(self, inputs, labels):

        inputs = self.augment_inputs(inputs)
        labels = self.augment_labels(labels)

        return inputs, labels


augmenter = Augment(seed=42)

train_ds = train_ds.map(
    augmenter,
    num_parallel_calls=tf.data.AUTOTUNE
)

train_batches = train_ds.cache().shuffle(BUFFER_SIZE).prefetch(tf.data.AUTOTUNE)
test_batches = test_ds.cache().prefetch(tf.data.AUTOTUNE)


# ============================================================
# Loss Functions
# ============================================================

@tf.function
def attention_mse(y_true, y_pred):

    _y_true = y_true[y_true != 1.0]
    _y_pred = y_pred[y_true != 1.0]

    squared_difference = tf.square(_y_true - _y_pred)

    return tf.reduce_mean(squared_difference, axis=-1)


@tf.function
def attention_mae(y_true, y_pred):

    _y_true = y_true[y_true != 1.0]
    _y_pred = y_pred[y_true != 1.0]

    squared_difference = tf.abs(_y_true - _y_pred)

    return tf.reduce_mean(squared_difference, axis=-1)


# ============================================================
# Evaluation Metrics
# ============================================================

def emae(y_true, y_pred):

    mask = tf.not_equal(y_true, 1.0)

    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    return tf.reduce_mean(tf.abs(y_true_masked - y_pred_masked))


def emse(y_true, y_pred):

    mask = tf.not_equal(y_true, 1.0)

    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    return tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))


def nmae(y_true, y_pred):

    mask = tf.not_equal(y_true, 1.0)

    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    abs_error = tf.abs(y_true_masked - y_pred_masked)

    max_S = tf.reduce_max(y_true_masked)

    return tf.reduce_mean(abs_error) / (max_S + 1e-8)


def nmse(y_true, y_pred):

    mask = tf.not_equal(y_true, 1.0)

    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    sq_error = tf.square(y_true_masked - y_pred_masked)

    max_S = tf.reduce_max(y_true_masked)

    return tf.reduce_mean(sq_error) / ((max_S ** 2) + 1e-8)


def epsnr(y_true, y_pred):

    mask = tf.not_equal(y_true, 1.0)

    y_true_masked = tf.boolean_mask(y_true, mask)
    y_pred_masked = tf.boolean_mask(y_pred, mask)

    mse = tf.reduce_mean(tf.square(y_true_masked - y_pred_masked))

    psnr = 20 * tf.math.log(
        1.0 / tf.sqrt(mse + 1e-8)
    ) / tf.math.log(10.0)

    return psnr


# ============================================================
# Model Training
# ============================================================

model = MultiViewUNet()

model.compile(
    loss=attention_mse,
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    metrics=[
        attention_mae,
        emae,
        emse,
        nmae,
        nmse,
        epsnr
    ]
)

model_path = os.path.join(MODEL_PATH, f"{EXP_NAME}.weights.h5")

callbacks = [

    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=PATIENCE,
        restore_best_weights=True
    ),

    tf.keras.callbacks.ModelCheckpoint(
        model_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=True
    )
]


history = model.fit(
    train_batches,
    validation_data=test_batches,
    epochs=N_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)


# ============================================================
# Save Model
# ============================================================

timestamp = datetime.datetime.now().strftime('%b-%d-%I-%M%p')

_ = model(tf.random.normal([1, 768, 768, 3]))

model.load_weights(model_path)

model.save(os.path.join(MODEL_PATH, timestamp))


# ============================================================
# Prediction
# ============================================================

test_ds_unbatched = test_batches.unbatch()

pred_path = os.path.join(PRED_PATH, EXP_NAME, timestamp)

os.makedirs(pred_path, exist_ok=True)


for idx, (input, target) in enumerate(tqdm(test_ds_unbatched)):

    target = tf.squeeze(target).numpy()

    prediction = tf.squeeze(
        model.predict(
            tf.expand_dims(input, axis=0)
        )
    ).numpy()

    channel_sum = tf.expand_dims(
        tf.reduce_sum(target, axis=-1), axis=-1
    )

    white_mask = tf.reduce_all(
        tf.equal(channel_sum, 3.0), axis=-1
    )

    expanded_mask = tf.expand_dims(white_mask, axis=-1)

    expanded_mask = tf.tile(expanded_mask, [1, 1, 3])

    prediction = tf.where(
        expanded_mask,
        tf.ones_like(prediction),
        prediction
    )

    plt.figure(figsize=(7,7))
    plt.imshow(target)
    plt.axis("off")
    plt.savefig(os.path.join(pred_path,f"{idx}_T.png"))
    plt.close()

    plt.figure(figsize=(7,7))
    plt.imshow(prediction)
    plt.axis("off")
    plt.savefig(os.path.join(pred_path,f"{idx}_P.png"))
    plt.close()


# ============================================================
# Training Curves
# ============================================================

try:

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8,6))

    plt.plot(loss,label='Training Loss')
    plt.plot(val_loss,label='Validation Loss')

    plt.legend()
    plt.ylabel('Loss')
    plt.title('Training & Validation Loss')

    plt.tight_layout()

    plt.savefig(os.path.join(MODEL_PATH,timestamp+".png"))

    plt.close()

except:

    print("Model did not finish training")

    model.evaluate(test_batches)