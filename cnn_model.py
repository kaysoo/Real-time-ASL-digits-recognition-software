from keras import Sequential
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import itertools

batch_size = 32
img_height = 224
img_width = 224

# splitting the dataset for training and testing
train_dir = "Gestures/Images"
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.1,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

validation_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.09,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

test_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    validation_split=0.01,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)
class_names = train_ds.class_names

# CNN model
model = Sequential([
    # first convolution layer
    layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
    # layers.GaussianNoise(0.2),
    # second convolution layer
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.Dropout(0.25),
    layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
    # third convolution layer
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
    # layers.Dropout(0.25),
    # fourth convolution layer
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
    # fifth convolution layer
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
    # sixth convolutional layer
    layers.Conv2D(128, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=2, strides=2, padding='same'),
    # layers.Dropout(0.25),
    # flattening
    layers.Flatten(),
    # full connection
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(class_names), activation='softmax')
])
# compiling the CNN
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
# fitting the model
history = model.fit(train_ds, batch_size=32, validation_batch_size=32, epochs=10)
# saving the model
model.save('cnn_model4.h5')

# Display and plotting of the accuracy and the loss values
plt.figure()
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['loss'], label='training loss')
plt.title('Loss/accuracy')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.show()

# testing model on test images
actual = []
pred = []
for images, labels in test_ds:
    for i in range(0, len(images)):
        image = images[i]
        image = np.expand_dims(image, axis=0)
        result = model.predict(image)
        pred.append(class_names[np.argmax(result)])
        actual.append(class_names[labels[i].numpy()])
model.evaluate(test_ds)


# drawing and displaying the Confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.figure(figsize=(15, 15))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


cm = confusion_matrix(y_true=actual, y_pred=pred)
plot_confusion_matrix(cm=cm, classes=class_names, title='Confusion Matrix')
plt.show()
