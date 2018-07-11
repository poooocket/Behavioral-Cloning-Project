import csv
import cv2
import numpy as np
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Read the data and save them in the list LINES  
def load_data(data_path, additional_data_path):
    lines = []
    """
    with open(data_path) as f1:
        file1 = csv.reader(f1)
        for i, line in enumerate(file1):
            if i > 0:
                for j, path in enumerate(line[:3]):
                    new_path = 'data/'+ str(path).lstrip()
                    line[j] = new_path                
                lines.append(line)
    """
    with open(additional_data_path) as f2:
        file2 = csv.reader(f2)
        for i, line in enumerate(file2):
            #if i < 30:
            for j, path in enumerate(line[:3]):
                new_path = path[92:]
                line[j] = new_path
            lines.append(line)
    return lines

# Extract the image file path from LINES and read images
def read_images(data):
    images = []
    for row in data:
        for i in row[:3]:
            image = cv2.imread(str(i))
            images.append(image)
    return images

# Save angles
# Add a correction to the image from left camera, and minus a correction to the image from right camera
def read_measurements(data):
    measurements = []
    correction = 0.21
    for row in data:
        steering_c = float(row[3])
        steering_l = steering_c + correction
        steering_r = steering_c - correction
        measurements.append(steering_c)
        measurements.append(steering_l)
        measurements.append(steering_r)
    return np.array(measurements)

# Grayscale image and improving the image contrast
def image_processing(images):
    new_images = []
    for image in images:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        equ = cv2.equalizeHist(gray)
        new_images.append(np.reshape(equ, (160, 320, 1)))
    return np.array(new_images)

# Flip images and angles
def flipping(images, measurements):
    images_flipped = []
    for image in images:
        image_flipped = cv2.flip(image, 1)
        images_flipped.append(np.reshape(image_flipped, (160, 320, 1)))
    measurements_flipped = - measurements
    return np.array(images_flipped), measurements_flipped

# Combine images and flipped images
data = load_data('data/driving_log.csv', 'new_data/driving_log.csv')
images = read_images(data)
images = image_processing(images)
measurements = read_measurements(data)
images_flipped, measurements_flipped = flipping(images, measurements)

X_train = np.concatenate((images, images_flipped), axis=0)
y_train = np.concatenate((measurements, measurements_flipped), axis=0)

# Build the model
model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(160, 320, 1)))
model.add(Cropping2D(((70,25), (0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

# Compile the model and train it
model.compile(optimizer='adam', loss='mse')
history_object = model.fit(X_train, y_train, validation_split=0.2, shuffle=True, batch_size=128, nb_epoch=2, verbose=1)
model.save('model.h5')

print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('Model Mean Squared Error Loss')
plt.ylabel('Mean Squared Error Loss')
plt.xlabel('Epoch')
plt.legend(['Training Set', 'Validation Set'], loc='upper right')
plt.show()



