import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import ast
from collections import Counter
from sklearn.utils import class_weight
from imblearn.over_sampling import RandomOverSampler

# Constants (some will be set dynamically)
BATCH_SIZE = 8
EPOCHS = 20
N_FRAMES = 20
FEATURE_DIM = 132  # Fixed based on MediaPipe output

def load_csv_data(csv_path, n_frames):
    df = pd.read_csv(csv_path)
    total_frames = len(df)
    data = np.zeros((total_frames, FEATURE_DIM), dtype=np.float32)
    for i in range(total_frames):
        frame_data = []
        for col in df.columns:
            parsed = ast.literal_eval(df[col].iloc[i])
            frame_data.extend(parsed)
        data[i] = frame_data[:FEATURE_DIM]
    
    if total_frames < n_frames:
        padded_data = np.zeros((n_frames, FEATURE_DIM), dtype=np.float32)
        padded_data[:total_frames] = data
        return padded_data
    
    if total_frames > n_frames:
        indices = np.linspace(0, total_frames - 1, n_frames, dtype=int)
        return data[indices]
    return data

def augment_data(frames, noise_factor=0.15, warp_factor=0.2):
    noise = np.random.normal(0, noise_factor, frames.shape)
    frames = frames + noise
    if random.random() > 0.5:
        num_warped = int(N_FRAMES * (1 + warp_factor * (random.random() - 0.5)))
        warp_indices = np.linspace(0, N_FRAMES-1, num_warped)
        warp_indices = np.clip(warp_indices, 0, N_FRAMES-1).astype(int)
        warped_frames = frames[warp_indices]
        if len(warped_frames) != N_FRAMES:
            x = np.linspace(0, len(warped_frames)-1, N_FRAMES)
            warped_frames = np.array([np.interp(x, np.arange(len(warped_frames)), warped_frames[:, i]) 
                                    for i in range(FEATURE_DIM)]).T
    else:
        warped_frames = frames
    return warped_frames

class FrameGenerator:
    def __init__(self, files_list, n_frames, training=False):
        self.files_list = files_list
        self.n_frames = n_frames
        self.training = training
        self.class_names = sorted(set(f.parent.name for f, _ in files_list))
        self.class_ids_for_name = {name: idx for idx, name in enumerate(self.class_names)}

    def get_class_distribution(self):
        return Counter(class_name for _, class_name in self.files_list)

    def __call__(self):
        files = self.files_list.copy()
        if self.training:
            random.shuffle(files)
        for file_path, class_name in files:
            frames = load_csv_data(file_path, self.n_frames)
            if self.training:
                frames = augment_data(frames)
            class_id = self.class_ids_for_name[class_name]
            yield frames, class_id

def build_model(num_classes):
    inputs = tf.keras.layers.Input(shape=(N_FRAMES, FEATURE_DIM))
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.LSTM(32, recurrent_dropout=0.3)(x)
    x = tf.keras.layers.Dense(32, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.05))(x)
    x = tf.keras.layers.Dropout(0.7)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def split_dataset(base_path, train_split=0.7, test_split=0.15, val_split=0.15):
    base_path = Path(base_path)
    all_files = []
    class_dict = {}
    for class_dir in base_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            files = [(f, class_name) for f in class_dir.iterdir() if f.is_file() and f.suffix == '.csv']
            class_dict[class_name] = files
            all_files.extend(files)
    
    # Dynamically set NUM_CLASSES based on the number of classes found
    num_classes = len(class_dict)
    print(f"Detected {num_classes} classes in the dataset at {base_path}")

    # Ensure at least one sample per class in val/test
    train_files = []
    val_files = []
    test_files = []
    for class_name, files in class_dict.items():
        random.shuffle(files)
        n = len(files)
        val_n = max(1, int(n * val_split))  # At least 1 for val
        test_n = max(1, int(n * test_split))  # At least 1 for test
        train_n = n - val_n - test_n
        if train_n < 1:  # Ensure train has at least 1
            train_n = 1
            val_n = max(1, n - train_n - test_n)
            test_n = n - train_n - val_n
        train_files.extend(files[:train_n])
        test_files.extend(files[train_n:train_n + test_n])
        val_files.extend(files[train_n + test_n:train_n + test_n + val_n])
    
    random.shuffle(train_files)
    random.shuffle(test_files)
    random.shuffle(val_files)
    return train_files, test_files, val_files, num_classes

def oversample_minority(files_list):
    file_paths = [f[0] for f in files_list]
    labels = [f[1] for f in files_list]
    ros = RandomOverSampler(random_state=42)
    file_paths_resampled, labels_resampled = ros.fit_resample(np.array(file_paths).reshape(-1, 1), labels)
    return [(Path(f[0]), l) for f, l in zip(file_paths_resampled, labels_resampled)]

def prepare_dataset(files_list, batch_size, n_frames, training=False):
    gen = FrameGenerator(files_list, n_frames, training=training)
    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(n_frames, FEATURE_DIM), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int64)
        )
    ).repeat()
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_size_inches(18.5, 10.5)
    ax1.plot(history.history['loss'], label='train')
    ax1.plot(history.history['val_loss'], label='validation')
    ax1.set_title('Loss')
    ax1.legend()
    ax2.plot(history.history['accuracy'], label='train')
    ax2.plot(history.history['val_accuracy'], label='validation')
    ax2.set_title('Accuracy')
    ax2.legend()
    plt.show()

# Main execution
base_path = 'D:\sinhala\Dataset - MP - CSV\Greetings'  # Change this to any subset or full dataset path
train_files, test_files, val_files, NUM_CLASSES = split_dataset(base_path)

train_files = oversample_minority(train_files)

train_gen = FrameGenerator(train_files, N_FRAMES)
val_gen = FrameGenerator(val_files, N_FRAMES)
test_gen = FrameGenerator(test_files, N_FRAMES)
print("Training class distribution:", train_gen.get_class_distribution())
print("Validation class distribution:", val_gen.get_class_distribution())
print("Testing class distribution:", test_gen.get_class_distribution())

train_dataset = prepare_dataset(train_files, BATCH_SIZE, N_FRAMES, training=True)
val_dataset = prepare_dataset(val_files, BATCH_SIZE, N_FRAMES)
test_dataset = prepare_dataset(test_files, BATCH_SIZE, N_FRAMES)

print(f"Training samples: {len(train_files)}")
print(f"Testing samples: {len(test_files)}")
print(f"Validation samples: {len(val_files)}")

class_names = train_gen.class_names
train_labels = [train_gen.class_ids_for_name[class_name] for _, class_name in train_files]
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_labels),
    y=train_labels
)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights:", class_weights_dict)

model = build_model(NUM_CLASSES)
model.summary()

steps_per_epoch = max(1, len(train_files) // BATCH_SIZE)
validation_steps = max(1, len(val_files) // BATCH_SIZE)

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint('sinhala_sign_subset.weights.h5',  # Generic name for any subset
                             monitor='val_accuracy',
                             save_best_only=True,
                             save_weights_only=True)

history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=EPOCHS,
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    class_weight=class_weights_dict,
                    callbacks=[early_stopping, checkpoint])

best_train_acc = max(history.history['accuracy'])
best_val_acc = max(history.history['val_accuracy'])
best_train_loss = min(history.history['loss'])
best_val_loss = min(history.history['val_loss'])
print(f'Best Training Accuracy: {best_train_acc:.4f}')
print(f'Best Validation Accuracy: {best_val_acc:.4f}')
print(f'Best Training Loss: {best_train_loss:.4f}')
print(f'Best Validation Loss: {best_val_loss:.4f}')

# Evaluate on test set
test_loss, test_accuracy = model.evaluate(test_dataset, steps=len(test_files) // BATCH_SIZE)
print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_loss:.4f}')

plot_history(history)
model.save('sinhala_sign_subset_full_Greetings.keras')  # Generic name for any subset