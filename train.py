import os
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Config 
TRAIN_DIR  = "data/Dataset/train"
TEST_DIR   = "data/Dataset/test"
IMG_SIZE   = (64, 64)
BATCH_SIZE = 16
EPOCHS     = 5
MODEL_PATH = "model/classifier.keras"

os.makedirs("model", exist_ok=True)

# Data Generators 
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    zoom_range=0.1,
    rotation_range=10
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary"
)

print("\n Class indices:", train_gen.class_indices)
# Expected: {'airplanes': 0, 'cars': 1}

# CNN Model 
model = models.Sequential([
    layers.Input(shape=(*IMG_SIZE, 3)),

    layers.Conv2D(16, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Train 
print("\n Starting training...\n")
history = model.fit(
    train_gen,
    validation_data=test_gen,
    epochs=EPOCHS
)

# Evaluate
loss, acc = model.evaluate(test_gen)
print(f"\n Test Accuracy : {acc * 100:.2f}%")
print(f" Test Loss     : {loss:.4f}")

# Save
model.save(MODEL_PATH)
print(f"\n Model saved to {MODEL_PATH}")
print(f" Class indices : {train_gen.class_indices}")