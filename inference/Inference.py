"""
MHUNet Inference and Evaluation Script

This script performs inference using a trained MultiViewUNet model
to predict TAWSS maps from curvature-based geometry images and
evaluates prediction performance.

"""

# ============================================================
# Imports
# ============================================================

import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt

from networks import MultiViewUNet
from skimage.metrics import structural_similarity as ssim
from scipy.stats import pearsonr

matplotlib.use('Agg')
plt.rcParams["font.size"] = 16


# ============================================================
# Configuration
# ============================================================

PROBLEM = "CURVATURE_2_TAWSS"
MODEL_NAME = "MHUNet_CURVATURE_TAWSS"

# ============================================================
# Paths
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "data", "images")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
PRED_DIR    = os.path.join(BASE_DIR, "predictions")

TRAIN_DIR = "Train/"
TEST_DIR = "Inference/"

INPUT_DIR = PROBLEM.split("_2_")[0]
TARGET_DIR = PROBLEM.split("_2_")[1]

IMG_SIZE = 512
BATCH_SIZE = 8
BUFFER_SIZE = 100

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "Curvature_2_TAWSS_MHUNet_Run.weights.h5"
)

EXP_NAME = MODEL_NAME


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


trainX = load_data_from_dir(os.path.join(DATASET_DIR, TRAIN_DIR, INPUT_DIR))
trainY = load_data_from_dir(os.path.join(DATASET_DIR, TRAIN_DIR, TARGET_DIR))

testX = load_data_from_dir(os.path.join(DATASET_DIR, TEST_DIR, INPUT_DIR))
testY = load_data_from_dir(os.path.join(DATASET_DIR, TEST_DIR, TARGET_DIR))

train_ds = tf.data.Dataset.zip((trainX, trainY))
test_ds = tf.data.Dataset.zip((testX, testY))


# ============================================================
# Normalization
# ============================================================

normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), normalization_layer(y)))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), normalization_layer(y)))

AUTOTUNE = tf.data.AUTOTUNE

train_batches = train_ds.cache().shuffle(BUFFER_SIZE).prefetch(buffer_size=AUTOTUNE)
test_batches = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ============================================================
# Global TAWSS Range Extraction
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

global_min = np.inf
global_max = -np.inf

for folder_num in range(2, 30):

    folder_path = os.path.join(base_dir, str(folder_num))

    if os.path.exists(folder_path):

        date_folders = [
            f for f in os.listdir(folder_path)
            if os.path.isdir(os.path.join(folder_path, f))
        ]

        for date_folder in date_folders:

            wss_path = os.path.join(folder_path, date_folder, 'wss')

            if os.path.exists(wss_path):

                csv_file = os.path.join(
                    wss_path,
                    'hemodynamics_tawss_osi_rrt_ecap.csv'
                )

                if os.path.exists(csv_file):

                    try:

                        df = pd.read_csv(csv_file)

                        TAWSS_cols = [
                            col for col in df.columns
                            if 'tawss' in col.lower()
                        ]

                        for col in TAWSS_cols:

                            global_min = min(global_min, df[col].min())
                            global_max = max(global_max, df[col].max())

                    except Exception as e:

                        print(f"Error reading {csv_file}: {e}")


print(f"Global TAWSS range: {global_min} ~ {global_max}")


# ============================================================
# Load Model
# ============================================================

model = MultiViewUNet()

model.build((None, IMG_SIZE, IMG_SIZE, 3))

model.load_weights(MODEL_PATH)


# ============================================================
# Metric Calculation
# ============================================================

def calc_metrics(y_true, y_pred, max_v):

    mask = ~(np.all(y_true == 1.0, axis=-1))

    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]

    mae = np.mean(np.abs(y_true_masked - y_pred_masked))
    mse = np.mean((y_true_masked - y_pred_masked) ** 2)

    rmse = np.sqrt(mse)

    nmae = mae / (max_v + 1e-8)
    nmse = mse / ((max_v ** 2) + 1e-8)
    nrmse = rmse / (max_v + 1e-8)

    psnr = 20 * np.log10(max_v / np.sqrt(mse)) if mse > 0 else float('inf')

    y_mean = np.mean(y_true_masked)
    ss_res = np.sum((y_true_masked - y_pred_masked) ** 2)
    ss_tot = np.sum((y_true_masked - y_mean) ** 2)

    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    return mae, mse, rmse, nmae, nmse, nrmse, psnr, r2


# ============================================================
# Inference
# ============================================================

timestamp = datetime.datetime.now().strftime('%b-%d-%H-%M%p')

pred_path = os.path.join(PRED_DIR, EXP_NAME, timestamp)

os.makedirs(pred_path, exist_ok=True)


emae = emse = ermse = nmae = nmse = nrmse = epsnr = 0

ssim_total = 0
pearson_total = 0
r2_total = 0


for idx, (input, target) in enumerate(test_batches.unbatch()):

    prediction = tf.squeeze(
        model.predict(tf.expand_dims(input, axis=0), verbose=0)
    ).numpy()

    prediction[tf.math.reduce_all((target == [1.0, 1.0, 1.0]), axis=-1)] = [1.0, 1.0, 1.0]

    target = target.numpy()

    # Convert normalized values back to physical scale
    target_tawss = target * (global_max - global_min) + global_min
    tawss_final = prediction * (global_max - global_min) + global_min

    tawss_final = np.clip(tawss_final, 0, None)

    # ROI mask
    roi_mask = ~np.all(input.numpy() == 1.0, axis=-1)

    # Grayscale conversion for SSIM
    gray_target = np.mean(target, axis=-1)
    gray_pred = np.mean(prediction, axis=-1)

    gray_target_roi = gray_target[roi_mask]
    gray_pred_roi = gray_pred[roi_mask]

    if np.std(gray_target_roi) > 0 and np.std(gray_pred_roi) > 0:

        ssim_val = ssim(
            gray_target_roi,
            gray_pred_roi,
            data_range=gray_target_roi.max() - gray_target_roi.min()
        )

        pearson_val, _ = pearsonr(
            gray_target_roi.flatten(),
            gray_pred_roi.flatten()
        )

    else:

        ssim_val = 0
        pearson_val = 0

    ssim_total += ssim_val
    pearson_total += pearson_val

    m_emae, m_emse, m_ermse, m_nmae, m_nmse, m_nrmse, m_epsnr, m_r2 = calc_metrics(
        target,
        prediction,
        global_max
    )

    emae += m_emae
    emse += m_emse
    ermse += m_ermse
    nmae += m_nmae
    nmse += m_nmse
    nrmse += m_nrmse
    epsnr += m_epsnr

    r2_total += m_r2

    N = idx + 1

    # Save images
    plt.figure(figsize=(7,7))
    plt.imshow(target, vmin=global_min, vmax=global_max, cmap='jet')
    plt.axis("off")
    plt.savefig(os.path.join(pred_path,f"{idx}_T.png"))
    plt.close()

    plt.figure(figsize=(7,7))
    plt.imshow(prediction, vmin=global_min, vmax=global_max, cmap='jet')
    plt.axis("off")
    plt.savefig(os.path.join(pred_path,f"{idx}_P.png"))
    plt.close()


# ============================================================
# Print Final Metrics
# ============================================================

print(f"EMAE: {emae / N:.5f}")
print(f"EMSE: {emse / N:.5f}")
print(f"ERMSE: {ermse / N:.5f}")
print(f"NMAE: {nmae / N:.5f}")
print(f"NMSE: {nmse / N:.5f}")
print(f"NRMSE: {nrmse / N:.5f}")
print(f"EPSNR: {epsnr / N:.2f} dB")

print(f"Avg SSIM: {ssim_total / N:.5f}")
print(f"Avg Pearson R: {pearson_total / N:.5f}")
print(f"Avg R²: {r2_total / N:.5f}")