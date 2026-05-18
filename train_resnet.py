import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import ResNet50
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image

# --- 1. CONFIGURATION ---
BASE_PATH = "dataset" 
TRAIN_PATH = os.path.join(BASE_PATH, "train")
TEST_PATH = os.path.join(BASE_PATH, "test")

BATCH_SIZE = 32
IMG_SIZE = (224, 224) 
EPOCHS = 10

# --- 2. ROBUST DATA LOADING USING PIL ---
# (Same as before to prevent channel errors)
def load_dataset_manual(directory):
    images = []
    labels = []
    class_names = sorted(os.listdir(directory))
    
    print(f"Scanning {directory}...")
    for label_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        count = 0
        for filename in os.listdir(class_dir):
            file_path = os.path.join(class_dir, filename)
            try:
                img = Image.open(file_path)
                img = img.convert('RGB') # Force RGB
                img = img.resize(IMG_SIZE)
                img_array = np.array(img)
                
                images.append(img_array)
                labels.append(label_idx)
                count += 1
            except Exception as e:
                print(f"Skipping corrupt file: {filename} - {e}")
        
        print(f"  Loaded {count} images for class: {class_name}")

    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    return dataset, class_names

# Load Train and Test
train_dataset, class_names = load_dataset_manual(TRAIN_PATH)
test_dataset, _ = load_dataset_manual(TEST_PATH)

num_classes = len(class_names)
print(f"Classes found: {class_names}")

# Split Train/Val
dataset_size = len(train_dataset)
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset = train_dataset.shuffle(buffer_size=1000, seed=123)
val_dataset = train_dataset.skip(train_size)
train_dataset = train_dataset.take(train_size)

train_dataset = train_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# --- 3. DATA AUGMENTATION ---
data_augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# --- 4. BUILD MODEL (RESNET50) ---
print("Building ResNet50 Model...")

# Load Pre-trained ResNet50
base_model = ResNet50(
    include_top=False, 
    weights='imagenet', 
    input_shape=(224, 224, 3)
)
base_model.trainable = False 

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)

# --- CRITICAL DIFFERENCE ---
# ResNet requires specific preprocessing (zero-centering), not the -1 to 1 scaling used by EfficientNet.
# We use a Lambda layer to apply the ResNet preprocessing function.
x = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input)(x)

x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x) 
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs, outputs)

# --- 5. COMPILE ---
model.compile(
    optimizer=optimizers.Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 6. TRAIN ---
print("Starting Training...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS
)

# --- 7. EVALUATE ---
print("Evaluating on Test Set...")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# --- 8. PLOT ---
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs_range = range(EPOCHS)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# --- 9. SAVE ---
model.save("waste_resnet_3class.keras")
print("Model saved as waste_resnet_3class.keras")