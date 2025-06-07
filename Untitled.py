#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install -U tensorflow-addons


# In[2]:


get_ipython().system('nvidia-smi')


# In[3]:


get_ipython().system('pip install visualkeras')
import os
import warnings
import itertools
import cv2
import seaborn as sns
import pandas as pd
import numpy  as np
from PIL import Image
from sklearn.utils import class_weight
from sklearn.metrics import confusion_matrix, classification_report
from collections import Counter

import tensorflow as tf
import tensorflow_addons as tfa
import visualkeras
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import plot_model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from sklearn.model_selection   import train_test_split
from keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


# General parameters
epochs = 15
image_size = 240
np.random.seed(42)
tf.random.set_seed(42)


# In[5]:


import os
print(os.getcwd())  # Shows current directory


# In[6]:


import os
import zipfile

# Set up paths (Modify according to your local Jupyter directory)
zip_file_path = "Concrete Crack Images for Classification.zip"  # If the file is in the same directory as the notebook
extract_path = "concrete-and-pavement-crack-images-1"

# Extract the ZIP file
if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ Dataset extracted successfully!")

# Check extracted files and folders
print("Extracted Files:", os.listdir(extract_path))


# In[7]:


folder_path = ("concrete-and-pavement-crack-images-1")


# In[8]:


import os
import cv2
import numpy as np

def load_and_process_dataset(folder_path):
    """Loads and processes images using alternative while loop structures."""

    dataset = []
    labels = []
    class_folders = ['Negative', 'Positive']

    class_index = 0
    while class_index < len(class_folders):
        class_folder = class_folders[class_index]
        images_path = os.path.join(folder_path, class_folder)

        image_index = 0
        while True:  # Loop infinitely until a "break" occurs
            try:
                image_name = os.listdir(images_path)[image_index]  # Access by index
                image_path = os.path.join(images_path, image_name)

                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                image = cv2.resize(image, (240, 240))

                dataset.append(image)
                labels.append(class_index)  # Use class_index directly for labels

                image_index += 1
            except IndexError:  # Handle end of image list
                break  # Exit the inner loop

        class_index += 1

    return np.array(dataset), np.array(labels)

# Assuming `folder_path` is already a string
dataset, labels = load_and_process_dataset(folder_path)

# Convert to NumPy arrays
dataset = np.array(dataset)
lab = np.array(labels)

# Print shapes
print(dataset.shape, labels.shape)


# In[9]:


dataset = np.array(dataset)
lab = np.array(lab)
print(dataset.shape, lab.shape)


# In[10]:


x_train, x_test, y_train, y_test = train_test_split(dataset, lab, test_size=0.2, shuffle=True, random_state=42)


# In[11]:


import matplotlib.pyplot as plt

def plot_state(state):
    images = [load_img(os.path.join(folder_path, state, img_name), target_size=(image_size, image_size))
              for img_name in os.listdir(os.path.join(folder_path, state))[:9]]

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for ax, img in zip(axes.flat, images):
        ax.imshow(img)
    plt.show()


# In[12]:


plot_state('Positive')


# In[13]:


pic_size = 240  # Set to the required image size


# In[14]:


model = tf.keras.Sequential([
    
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2), activation="relu", padding="valid",input_shape=(pic_size,pic_size,3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(filters=32,kernel_size=(3,3),strides=(2,2), activation="relu", padding="valid"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=64, activation='relu', 
                          kernel_regularizer=regularizers.L1L2(l1=1e-3, l2=1e-3), 
                          bias_regularizer=regularizers.L2(1e-2),
                          activity_regularizer=regularizers.L2(1e-3)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),
])


# In[15]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()


# In[16]:


plot_model(model, show_shapes=True, show_layer_names=False)


# In[17]:


visualkeras.layered_view(model, legend=True)


# In[18]:


class_weights = class_weight.compute_class_weight(class_weight = "balanced", classes= np.unique(y_train), y= y_train)
class_weights = dict(zip(np.unique(y_train), class_weights))
class_weights


# In[19]:


# Define a directory to save the model
save_directory = "road"  # Change this path if needed

# Save the trained model
model.save(save_directory)
print(f"Model saved successfully in: {save_directory}")


# In[57]:


import pickle

# Save history after training
with open("history.pkl", "wb") as f:
    pickle.dump(vit_history.history, f)


# In[20]:


history = model.fit(x_train,y_train,epochs = 100, class_weight=class_weights, validation_data=(x_test, y_test),verbose=1)


# In[59]:


import pickle

# Save the history object
with open("history.pkl", "wb") as f:
    pickle.dump(history.history, f)


# In[60]:


import json

# Save history in JSON format
with open("history.json", "w") as f:
    json.dump(history.history, f)


# In[61]:


with open("history.pkl", "rb") as f:
    history_data = pickle.load(f)

# Example: Plot accuracy
import matplotlib.pyplot as plt

plt.plot(history_data["accuracy"], label="Training Accuracy")
plt.plot(history_data["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.show()


# In[21]:


# Define a directory to save the model
save_directory = "road"  # Change this path if needed

# Save the trained model
model.save(save_directory)
print(f"Model saved successfully in: {save_directory}")


# In[22]:


from tensorflow.keras.models import load_model

# Load the saved model
loaded_model = load_model("road")

# Verify the model summary
loaded_model.summary()


# In[23]:


import os
print(os.listdir("road"))


# In[24]:


model.save("road.h5")
loaded_model = load_model("road.h5")


# In[25]:


plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[26]:


results = model.evaluate(x_test, y_test)
print('The current model achieved an accuracy of {}%!'.format(round(results[1]*100,2)))


# In[64]:


from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Convert true labels (assuming y_test is binary)
y_true = y_test  # Ensure y_test is in correct format

# Compute metrics
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Print classification report
print(classification_report(y_true, y_pred))

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")


# In[27]:


# compute predictions
predictions = model.predict(x_test)
y_pred = []
for i in predictions:
    if i >= 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)


# In[28]:


predictions = model.predict(x_test)
y_pred = (predictions >= 0.5).astype(int)


# In[29]:


import matplotlib.pyplot as plt
import seaborn as sns

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# Normalize for better visualization
cnf_matrix_norm = cnf_matrix.astype('float') / cnf_matrix.sum(axis=1)[:, np.newaxis]

# Employ a triangular structure for visual clarity
mask = np.triu(np.ones_like(cnf_matrix_norm, dtype=bool))

# Create a visually distinct heatmap with a different colormap
sns.heatmap(cnf_matrix_norm, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', vmin=0, vmax=1,
            linewidths=0.5, cbar_kws={'label': 'Normalized Values'})

# Customize labels and title
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (Triangular Structure)')

plt.show()


# In[30]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# Create a hexagonal mask for a visually striking shape
mask = np.triu(np.ones_like(cnf_matrix, dtype=bool)) & np.tril(np.ones_like(cnf_matrix, dtype=bool), k=-1)

# Plot using Seaborn with distinct customizations
sns.heatmap(
    cnf_matrix,
    annot=True,
    fmt='.2f',
    cmap='cubehelix',  # Employ a vibrant colormap
    mask=mask,
    linewidths=0.5,
    cbar=False,
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.title('Confusion Matrix - Hexagonal Emphasis')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[31]:


learning_rate = 0.001
weight_decay = 0.0001
batch_size = 256
num_epochs = 100
image_size = 240  # We'll resize input images to this size
patch_size = 20  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 64
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier


# In[32]:


data_augmentation = tf.keras.Sequential(
    [
        layers.Normalization(),
        layers.Resizing(image_size, image_size),
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(factor=0.02),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2
        ),
    ],
    name="data_augmentation",
)
# Compute the mean and the variance of the training data for normalization.
data_augmentation.layers[0].adapt(x_train)


# In[33]:


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x


# In[34]:


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches


# In[37]:


plt.figure(figsize=(8, 8))
image = x_train[np.random.choice(range(x_train.shape[0]))]
plt.imshow(image.astype("uint8"))

resized_image = tf.image.resize(
    tf.convert_to_tensor([image]), size=(image_size, image_size)
)
patches = Patches(patch_size)(resized_image)
print(f"Image size: {image_size} X {image_size}")
print(f"Patch size: {patch_size} X {patch_size}")
print(f"Patches per image: {patches.shape[1]}")
print(f"Elements per patch: {patches.shape[-1]}")

n = int(np.sqrt(patches.shape[1]))
plt.figure(figsize=(8, 8))
for i, patch in enumerate(patches[0]):
    ax = plt.subplot(n, n, i + 1)
    patch_img = tf.reshape(patch, (patch_size, patch_size, 3))
    plt.imshow(patch_img.numpy().astype("uint8"))
    plt.axis("off")


# In[39]:


class PatchEncoder(tf.keras.layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded


# In[40]:


def create_vit_classifier():
    inputs = layers.Input(shape=(240, 240, 3))
    # Augment data.
    augmented = data_augmentation(inputs)
    # Create patches.
    patches = Patches(patch_size)(augmented)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = layers.Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = layers.Add()([x3, x2])

    # Create a [batch_size, projection_dim] tensor.
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.Flatten()(representation)
    representation = layers.Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = layers.Dense(2)(features)
    # Create the Keras model.
    model = tf.keras.Model(inputs=inputs, outputs=logits)
    return model


# In[42]:


def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(5, name="top-5-accuracy"),
        ],
    )

    checkpoint_filepath = "models/vit_checkpoint"  # Changed directory
    os.makedirs(os.path.dirname(checkpoint_filepath), exist_ok=True)  # Ensure directory exists

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
    )

    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batch_size,
        epochs=10,  # Uses num_epochs from global variables
        validation_data=(x_test, y_test),
        callbacks=[checkpoint_callback],
    )

    model.load_weights(checkpoint_filepath)
    _, accuracy, top_5_accuracy = model.evaluate(x_test, y_test)
    print(f"✅ Test Accuracy: {round(accuracy * 100, 2)}%")
    print(f"✅ Test Top-5 Accuracy: {round(top_5_accuracy * 100, 2)}%")

    return history


# In[43]:


vit_classifier = create_vit_classifier()
vit_history = run_experiment(vit_classifier)


# In[52]:


vit_classifier.save("vit_model_1.h5")  # Saves the model in HDF5 format


# In[47]:


# Save the trained model in TensorFlow format
vit_classifier.save("vit_model.h5")  # Saves as an HDF5 file
# OR save it in the SavedModel format (recommended for TensorFlow serving)
vit_classifier.save("vit_model")


# In[ ]:





# In[44]:


plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(vit_history.history['loss'], label='Training Loss')
plt.plot(vit_history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(vit_history.history['accuracy'], label='Training Accuracy')
plt.plot(vit_history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()


# In[45]:


# compute predictions
vit_predictions = vit_classifier.predict(x_test)
vit_y_pred = [np.argmax(probas) for probas in vit_predictions]


# In[46]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)

# Create a hexagonal mask for a visually striking shape
mask = np.triu(np.ones_like(cnf_matrix, dtype=bool)) & np.tril(np.ones_like(cnf_matrix, dtype=bool), k=-1)

# Plot using Seaborn with distinct customizations
sns.heatmap(
    cnf_matrix,
    annot=True,
    fmt='.2f',
    cmap='cubehelix',  # Employ a vibrant colormap
    mask=mask,
    linewidths=0.5,
    cbar=False,
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"]
)
plt.title('Confusion Matrix - Hexagonal Emphasis')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[66]:


from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

# Use y_test directly
vit_y_true = y_test  
vit_y_pred = [np.argmax(probas) for probas in vit_predictions]  # Convert probabilities to class labels

# Compute metrics
precision = precision_score(vit_y_true, vit_y_pred, average="weighted")
recall = recall_score(vit_y_true, vit_y_pred, average="weighted")
f1 = f1_score(vit_y_true, vit_y_pred, average="weighted")

# Print classification report
print(classification_report(vit_y_true, vit_y_pred))

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

