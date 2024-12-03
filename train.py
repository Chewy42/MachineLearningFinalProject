import tensorflow as tf
from tensorflow.keras import layers, models
import json
import os
import numpy as np
from sklearn.model_selection import train_test_split

# Force GPU usage
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
tf.config.set_visible_devices(physical_devices[0], 'GPU')

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

config = load_config()
DATASET_PATH = config['paths']['dataset']
ANNOTATIONS_PATH = config['paths']['annotations']

@tf.function
def load_and_preprocess_image(img_path, target_size=(224, 224)):
    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, target_size)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def load_data(dataset_path, annotations_path):
    images = []
    labels = []
    for img_name in os.listdir(dataset_path):
        img_path = os.path.join(dataset_path, img_name)
        images.append(img_path)
        
        ann_path = os.path.join(annotations_path, img_name.replace('.jpg', '.txt'))
        with open(ann_path, 'r') as f:
            ann = f.read().strip()
        labels.append(int(ann))

    return np.array(images), np.array(labels)

def build_model(num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

X, y = load_data(DATASET_PATH, ANNOTATIONS_PATH)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_ds = train_ds.map(lambda x, y: (load_and_preprocess_image(x), y))
train_ds = train_ds.batch(64).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_ds = test_ds.map(lambda x, y: (load_and_preprocess_image(x), y))
test_ds = test_ds.batch(64).prefetch(tf.data.AUTOTUNE)

num_classes = len(np.unique(y))

with tf.device('/GPU:0'):
    model = build_model(num_classes)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(train_ds, epochs=100, 
                        validation_data=test_ds,
                        verbose=1)

    test_loss, test_acc = model.evaluate(test_ds, verbose=2)
    print(f'\nTest accuracy: {test_acc}')
