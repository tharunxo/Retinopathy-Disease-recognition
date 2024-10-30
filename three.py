import numpy as np
import pandas as pd
import cv2
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# List all files in the directory
for dirname, _, filenames in os.walk('F:/HCL/retina/input/ocular-disease-recognition-odir5k'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Read the CSV file
df = pd.read_csv("F:/HCL/retina/input/ocular-disease-recognition-odir5k/full_df.csv")

# Function to check if cataract or diabetic retinopathy is mentioned
def has_cataract(text):
    return 1 if "cataract" in text else 0

def has_diabetic(text):
    return 1 if "moderate" in text else 0 

# Apply the function to the relevant columns
df["left_cataract"] = df["Left-Diagnostic Keywords"].apply(lambda x: has_cataract(x))
df["right_cataract"] = df["Right-Diagnostic Keywords"].apply(lambda x: has_cataract(x))
df["left_diabetic"] = df["Left-Diagnostic Keywords"].apply(lambda x: has_diabetic(x))
df["right_diabetic"] = df["Right-Diagnostic Keywords"].apply(lambda x: has_diabetic(x))

# Filter images with cataract, diabetic retinopathy, and normal fundus
left_cataract = df.loc[(df.C == 1) & (df.left_cataract == 1)]["Left-Fundus"].values
right_cataract = df.loc[(df.C == 1) & (df.right_cataract == 1)]["Right-Fundus"].values
left_diabetic = df.loc[(df.D == 1) & (df.left_diabetic == 1)]["Left-Fundus"].values
right_diabetic = df.loc[(df.D == 1) & (df.right_diabetic == 1)]["Right-Fundus"].values
left_normal = df.loc[(df.C == 0) & (df.D == 0) & (df["Left-Diagnostic Keywords"] == "normal fundus")]["Left-Fundus"].sample(250, random_state=42).values
right_normal = df.loc[(df.C == 0) & (df.D == 0) & (df["Right-Diagnostic Keywords"] == "normal fundus")]["Right-Fundus"].sample(250, random_state=42).values

# Combine the images
cataract = np.concatenate((left_cataract, right_cataract), axis=0)
diabetic = np.concatenate((left_diabetic, right_diabetic), axis=0)
normal = np.concatenate((left_normal, right_normal), axis=0)

from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Image preprocessing
dataset_dir = "F:/HCL/retina/preprocessed_images"
image_size = 224
dataset = []

def create_dataset(image_category, label):
    for img in tqdm(image_category):
        image_path = os.path.join(dataset_dir, img)
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.resize(image, (image_size, image_size))
        except:
            continue
        
        dataset.append([np.array(image), np.array(label)])
    random.shuffle(dataset)
    return dataset


dataset = create_dataset(cataract, 1)
dataset = create_dataset(diabetic, 2)
dataset = create_dataset(normal, 0)


plt.figure(figsize=(12, 7))
for i in range(10):
    sample = random.choice(range(len(dataset)))
    image = dataset[sample][0]
    category = dataset[sample][1]
    if category == 1:
        label = "Cataract"
    elif category == 2:
        label = "Diabetic"
    else:
        label = "Normal"
    plt.subplot(2, 5, i + 1)
    plt.imshow(image)
    plt.xlabel(label)
plt.tight_layout()

x = np.array([i[0] for i in dataset]).reshape(-1, image_size, image_size, 3)
y = np.array([i[1] for i in dataset])
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from tensorflow.keras.applications.vgg19 import VGG19
vgg = VGG19(weights="imagenet", include_top=False, input_shape=(image_size, image_size, 3))
for layer in vgg.layers:
    layer.trainable = False

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

datagen.fit(x_train)

# Build the model with additional layers and dropout for regularization
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout

model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(3, activation="softmax"))

# Unfreeze the last few layers of VGG19 for fine-tuning
for layer in vgg.layers[-4:]:
    layer.trainable = True

from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(learning_rate=0.0001), loss="sparse_categorical_crossentropy", metrics=["accuracy"])

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint("vgg19_multiclass.keras", monitor="val_accuracy", verbose=1, save_best_only=True,
                             save_weights_only=False, save_freq='epoch')
earlystop = EarlyStopping(monitor="val_accuracy", patience=5, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)

# Train the model with data augmentation
history = model.fit(datagen.flow(x_train, y_train, batch_size=32), epochs=1, validation_data=(x_test, y_test),
                    verbose=1, callbacks=[checkpoint, earlystop, reduce_lr])

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print("Loss:", loss)
print("Accuracy:", accuracy)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

y_pred_prob = model.predict(x_test)
y_pred = np.argmax(y_pred_prob, axis=1)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

from mlxtend.plotting import plot_confusion_matrix
cm = confusion_matrix(y_test, y_pred)
plot_confusion_matrix(conf_mat=cm, figsize=(8, 7), class_names=["Normal", "Cataract", "Diabetic"], show_normed=True)
plt.show()
