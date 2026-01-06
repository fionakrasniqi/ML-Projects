import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing import image_dataset_from_directory

# Set these to the paths for your 'Training' and 'Testing' folders
TRAIN_DIR = r"C:\Users\Admin\Downloads\Brain Tumor\Training"
TEST_DIR  = r"C:\Users\Admin\Downloads\Brain Tumor\Testing"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 10

# Load training and validation datasets (80/20 split from Training/)
train_ds = image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="training",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
val_ds = image_dataset_from_directory(
    TRAIN_DIR,
    validation_split=0.2,
    subset="validation",
    seed=42,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names
train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
val_ds   = val_ds.prefetch(tf.data.AUTOTUNE)


test_ds = image_dataset_from_directory(
    TEST_DIR,
    shuffle=False,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

# Build a model using transfer learning (EfficientNetB0)
base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    pooling="avg",
)
inputs = tf.keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.Dense(256, activation="relu")(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)
model.save("brain_tumor_model.h5")

# Plot training/validation accuracy and loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train")
plt.plot(history.history["val_accuracy"], label="Val")
plt.title("Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train")
plt.plot(history.history["val_loss"], label="Val")
plt.title("Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on the test set
test_images = []
test_labels = []
for imgs, labels in test_ds:
    test_images.append(imgs.numpy())
    test_labels.append(labels.numpy())
test_images = np.concatenate(test_images)
test_labels = np.concatenate(test_labels)

pred_probs = model.predict(test_images)
pred_classes = np.argmax(pred_probs, axis=1)


print(classification_report(test_labels, pred_classes, target_names=class_names))
cm = confusion_matrix(test_labels, pred_classes)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()

# Display predictions for the first 10 test images
print("\nBelow you see random MRI scans from the test set.")
print("For each image we show:")
print("- True: the real class from the dataset")
print("- Pred: the model's predicted class")
print("- Conf: the model's confidence in that prediction (probability %)")

n_show = min(20, len(test_images))
idxs = np.random.choice(len(test_images), size=n_show, replace=False)

plt.figure(figsize=(18, 12))

rows = 4
cols = 5

for i, idx in enumerate(idxs):
    img = test_images[idx].astype("uint8")
    true_idx = test_labels[idx]
    pred_idx = pred_classes[idx]
    conf = pred_probs[idx, pred_idx] * 100

    plt.subplot(rows, cols, i + 1)
    plt.imshow(img)
    plt.axis("off")

    title = (
        f"True: {class_names[true_idx]}\n"
        f"Pred: {class_names[pred_idx]} ({conf:.1f}%)"
    )
    plt.title(title, fontsize=8)

plt.suptitle("Random Test MRIs with Model Predictions", fontsize=15, y=1.02)
plt.tight_layout()
plt.show()
