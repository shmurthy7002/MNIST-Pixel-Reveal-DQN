import struct
from os.path import join
import cupy as cp
import numpy as np


# -------------------------------------------------------------------
# 1. MNIST Dataloader
# -------------------------------------------------------------------
class MnistDataloader(object):

  def __init__(
      self,
      training_images_filepath,
      training_labels_filepath,
      test_images_filepath,
      test_labels_filepath,
  ):
    self.training_images_filepath = training_images_filepath
    self.training_labels_filepath = training_labels_filepath
    self.test_images_filepath = test_images_filepath
    self.test_labels_filepath = test_labels_filepath

  def read_images_labels(self, images_filepath, labels_filepath):
    with open(labels_filepath, 'rb') as file:
      magic, size = struct.unpack('>II', file.read(8))
      if magic != 2049:
        raise ValueError(
            f'Magic number mismatch, expected 2049, got {magic}'
        )
      labels = np.frombuffer(file.read(), dtype=np.uint8)

    with open(images_filepath, 'rb') as file:
      magic, size, rows, cols = struct.unpack('>IIII', file.read(16))
      if magic != 2051:
        raise ValueError(
            f'Magic number mismatch, expected 2051, got {magic}'
        )
      images = np.frombuffer(file.read(), dtype=np.uint8).reshape(
          size, rows, cols
      )

    return images, labels

  def load_data(self):
    x_train, y_train = self.read_images_labels(
        self.training_images_filepath, self.training_labels_filepath
    )
    x_test, y_test = self.read_images_labels(
        self.test_images_filepath, self.test_labels_filepath
    )
    return (x_train, y_train), (x_test, y_test)


# -------------------------------------------------------------------
# 2. CuPy Masked Dataset Generator
# -------------------------------------------------------------------
def process_dataset_masked_cupy(
    x_cpu: np.ndarray, y_cpu: np.ndarray, top_k=120, select_k=60, repeats=3
):
  """Uses GPU via CuPy to:

  1. Find top `top_k` brightest pixels for each image. 2. Randomly select
  `select_k` of them `repeats` times. 3. Zero-mask the remaining 724 pixels,
  returning full 784-dimensional vectors.
  """
  N = x_cpu.shape[0]

  # Move arrays to GPU
  x_gpu = cp.asarray(x_cpu, dtype=cp.float32)  # Shape: (N, 784)
  y_gpu = cp.asarray(y_cpu, dtype=cp.int64)  # Shape: (N,)

  # --- Step A: Get indices of top 120 brightest pixels per image ---
  top_120_idx = cp.argpartition(x_gpu, -top_k, axis=1)[:, -top_k:]  # (N, 120)

  # --- Step B: Subsample 60 indices from 120 (3 times per image) ---
  rand_noise = cp.random.random((repeats, N, top_k), dtype=cp.float32)
  perm_relative = cp.argsort(rand_noise, axis=-1)[
      :, :, :select_k
  ]  # (3, N, 60)

  # Map relative [0..119] indices back to original [0..783] spatial indices
  top_120_expanded = cp.tile(top_120_idx[None, :, :], (repeats, 1, 1))
  chosen_784_idx = cp.take_along_axis(
      top_120_expanded, perm_relative, axis=-1
  )  # (3, N, 60)

  # Extract the original pixel values for these 60 indices
  x_gpu_expanded = cp.tile(x_gpu[None, :, :], (repeats, 1, 1))
  chosen_values = cp.take_along_axis(
      x_gpu_expanded, chosen_784_idx, axis=-1
  )  # (3, N, 60)

  # --- Step C: Create 784-pixel sparse output tensors ---
  # Initialize zero tensors (masked out background)
  out_gpu = cp.zeros((repeats, N, 784), dtype=cp.float32)

  # Place selected 60 pixel values back into their original spatial locations
  cp.put_along_axis(out_gpu, chosen_784_idx, chosen_values, axis=-1)

  # --- Step D: Format and group output samples ---
  # Group by original image: [img0_run1, img0_run2, img0_run3, img1_run1, ...]
  out_gpu = out_gpu.swapaxes(0, 1).reshape(N * repeats, 784)
  y_expanded = cp.repeat(y_gpu, repeats)

  # Move back to CPU NumPy
  return cp.asnumpy(out_gpu), cp.asnumpy(y_expanded)


# -------------------------------------------------------------------
# 3. Main Execution
# -------------------------------------------------------------------
input_path = ''
training_images_filepath = join(
    input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte'
)
training_labels_filepath = join(
    input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte'
)
test_images_filepath = join(
    input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte'
)
test_labels_filepath = join(
    input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte'
)

# Load raw dataset
mnist_dataloader = MnistDataloader(
    training_images_filepath,
    training_labels_filepath,
    test_images_filepath,
    test_labels_filepath,
)
(x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = (
    mnist_dataloader.load_data()
)

# Normalize raw inputs [0.0, 1.0]
x_train_flat = (
    np.array([x.ravel() for x in x_train_raw], dtype=np.float32) / 255.0
)
x_test_flat = (
    np.array([x.ravel() for x in x_test_raw], dtype=np.float32) / 255.0
)

print('Processing training set on GPU...')
x_train_new, y_train_new = process_dataset_masked_cupy(
    x_train_flat, y_train_raw
)

print('Processing test set on GPU...')
x_test_new, y_test_new = process_dataset_masked_cupy(x_test_flat, y_test_raw)

print(f'\nNew Train Dataset Shape: {x_train_new.shape}')  # (180000, 784)
print(f'New Test Dataset Shape:  {x_test_new.shape}')  # (30000, 784)

# Verify that exactly 60 non-zero pixels exist per sample
print(
    'Non-zero pixel count check (should be 60):',
    np.count_nonzero(x_train_new[0]),
)

# -------------------------------------------------------------------
# 4. Save to Disk
# -------------------------------------------------------------------
output_filename = 'mnist_784px_masked_60active.npz'
np.savez_compressed(
    output_filename,
    x_train=x_train_new,
    y_train=y_train_new,
    x_test=x_test_new,
    y_test=y_test_new,
)

print(f"\nDataset successfully generated and saved to '{output_filename}'!")