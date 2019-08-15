import os
import keras
from keras import layers
from keras import models
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Flatten, Dense, Dropout
from keras.optimizers import SGD
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy
import pandas as pd
import csv


def plt_modle(model_hist):
    acc = model_hist.history['acc']
    val_acc = model_hist.history['val_acc']
    loss = model_hist.history['loss']
    val_loss = model_hist.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, color='#0984e3', marker='o', linestyle='none', label='Training Accuracy')
    plt.plot(epochs, val_acc, color='#0984e3', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, color='#eb4d4b', marker='o', linestyle='none', label='Training Loss')
    plt.plot(epochs, val_loss, color='#eb4d4b', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='best')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    plt.show()

def loadWeights(fileName):
    return keras.models.load_model(fileName)


def predict(myModel, path):
    images = []
    names = []
    classifyNames = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
    for filename in os.listdir(path):
        if filename.__contains__(".jpg"):
            img = image.load_img(os.path.join(path, filename), target_size=(128, 128, 3), grayscale=False)
            img = image.img_to_array(img)
            img = img / 255
            images.append(img)
            names.append(filename)
    imagesArray = numpy.array(images)
    results = myModel.predict_classes(imagesArray)
    return writeResults(classifyNames, names, results)


def writeResults(classifyNames, imageNames, classifications):
    categories = []
    dictionaryResults = {}
    for i in range(len(imageNames)):
        categories.append(str(classifyNames[classifications[i]]))
        dictionaryResults[imageNames[i]] = categories[i]
    resultsFile = pd.DataFrame({"image_name": imageNames, "classify": categories})
    resultsFile.to_csv('results.csv', index=False)
    return dictionaryResults

def selectByCategory(categoryName, resultsPath):
    results = []
    with open(resultsPath) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['classify'] == categoryName:
                results.append(row['image_name'])

    return results

# Split images into Training and Validation Sets (20%)
train = ImageDataGenerator(rescale=1./255, horizontal_flip=True, shear_range=0.2, zoom_range=0.2, width_shift_range=0.2,
                           height_shift_range=0.2, fill_mode='nearest', validation_split=0.2)
img_size = 128
batch_size = 20
t_steps = 3462/batch_size
v_steps = 861/batch_size
classes = 5
flower_path = "flowers/"
train_gen = train.flow_from_directory(flower_path, target_size=(img_size, img_size), batch_size=batch_size,
                                      class_mode='categorical', subset='training')
valid_gen = train.flow_from_directory(flower_path, target_size=(img_size, img_size), batch_size=batch_size,
                                      class_mode='categorical', subset='validation')


# Model
model = models.Sequential()
model.add(Conv2D(96, (11, 11), strides=(4, 4), activation='relu', padding='same', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
# Local Response normalization for Original Alexnet
model.add(BatchNormalization())

model.add(Conv2D(256, (5, 5), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# Local Response normalization for Original Alexnet
model.add(BatchNormalization())
model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(384, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
# Local Response normalization for Original Alexnet
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dense(classes))
# last layer should be with softmax activation function - do not change!!!
model.add(layers.Dense(classes, activation='softmax'))

optimizer = SGD(lr=0.1, decay=0.1, momentum=0.1, nesterov=True)

loss = 'binary_crossentropy'
# model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

# model_hist = model.fit_generator(train_gen, steps_per_epoch=t_steps, epochs= 20 , validation_data=valid_gen,
# validation_steps=v_steps)
# model.save('flowers_model.h5')
# plt_model(model_hist)






