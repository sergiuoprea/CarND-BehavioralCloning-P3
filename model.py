#Imports
import os
import csv as csv
import numpy as np
import cv2

#sklearn
import sklearn
from sklearn.model_selection import train_test_split

#Keras imports
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Cropping2D, Dropout
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras import regularizers

img_paths = ['./data_own/Data_3_vueltas_delante/IMG/', './data_own/Data_3_vueltas_reves/IMG/','./data_own/Data_14_vueltas/IMG/' , './data_own/Recovery_laps/IMG/']
csv_paths = ['./data_own/Data_3_vueltas_delante/driving_log.csv', './data_own/Data_3_vueltas_reves/driving_log.csv','./data_own/Data_14_vueltas/driving_log.csv', './data_own/Recovery_laps/driving_log.csv']


########################################################
#################### Read data #########################
########################################################

#Function to read CSV file
def readCSV(in_csv_path, in_path):
	lines = []

	with open(in_csv_path) as csvfile:
		reader = csv.reader(csvfile)
		next(reader, None) #Skip header

		for line in reader:
			#Correct images paths
			line_aux = []
			#Center image
			line_aux.append(in_path + line[0].split('/')[-1])
			#Left image
			line_aux.append(in_path + line[1].split('/')[-1])
			#Right image
			line_aux.append(in_path + line[2].split('/')[-1])
			#Steering angle
			line_aux.append(float(line[3]))

			lines.append(line_aux)
	    
	return lines

#Function to load data using information from a given csv line
def readData_Advanced(in_line, preprocess, correction= 0.25):
	images = []
	measurements = []

	center_img = cv2.imread(in_line[0])
	left_img = cv2.imread(in_line[1])
	right_img = cv2.imread(in_line[2])

	if preprocess:
		center_img = preprocess_image(center_img)
		left_img = preprocess_image(left_img)
		right_img = preprocess_image(right_img)

	images.append(center_img)
	images.append(left_img)
	images.append(right_img)

	#Measurements
	center_steer = in_line[3]
	left_steer = center_steer + correction
	right_steer = center_steer - correction

	measurements.append(center_steer)
	measurements.append(left_steer)
	measurements.append(right_steer)

	return images, measurements


########################################################
############### Image processing #######################
########################################################

#Inspired on Jeremy Shannon implementation
def preprocess_image(in_image):
	new_img = cv2.cvtColor(in_image, cv2.COLOR_BGR2YUV)
	return new_img

########################################################
############### Data augmentation ######################
########################################################

#Function to augment the dataset. how_much indicates the % of how much data to augment
def augmentDataset(in_images, in_measurements, how_much= 0.3):
	augm_imgs = []
	augm_ang = []
	augmented_num = int(len(in_images) * how_much)

	sklearn.utils.shuffle(in_images, in_measurements)

	in_images = in_images[:augmented_num]
	in_measurements = in_measurements[:augmented_num]
    
	for img, ang in zip(in_images, in_measurements):
		augm_imgs.append(np.fliplr(img))
		augm_ang.append(-ang)
    
	return augm_imgs, augm_ang


########################################################
############### Model Architecture #####################
########################################################

#Inspired on Jeremy Shannon implementation
def NvidiaArchitecture():
        model = Sequential()

        #Preprocessing
        model.add(Lambda(lambda x: (x/ 255.0) - 0.5, input_shape=(160, 320, 3)))
        model.add(Cropping2D(cropping=((50,20), (0,0))))

        model.add(Conv2D(24,(5,5), strides=(2,2), activation='elu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Conv2D(36,(5,5), strides=(2,2), activation='elu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Conv2D(48,(5,5), strides=(2,2), activation='elu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Conv2D(64,(3,3), activation='elu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Conv2D(64,(3,3), activation='elu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Flatten())
        model.add(Dense(100, activation= 'elu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.3))
        model.add(Dense(50, activation= 'elu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.3))
        model.add(Dense(10, activation= 'elu', kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dropout(0.3))
        model.add(Dense(1))

        return model

########################################################
################## Generator ###########################
########################################################

def generator(samples, batch_size=32, data_augment=False, augment_factor=0.3, preprocess=False):
    num_samples = len(samples)

    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                aux_img, aux_ang = readData_Advanced(batch_sample, preprocess=preprocess)
                images.extend(aux_img)
                angles.extend(aux_ang)
   
            if data_augment:
                augm_img, augm_ang = augmentDataset(images, angles)
                images.extend(augm_img)
                angles.extend(augm_ang)
                
            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


########################################################
################## Execution ###########################
########################################################

#Variables
batch_size= 128
max_epochs = 200
augment_factor = 0.5
early_stopping_patience = 3

#Load datasets paths to images and steering angle value parsed to float
samples = []
for img_path, csv_path in zip(img_paths, csv_paths):
	samples.extend(readCSV(csv_path, img_path))

#Splitting data into training and validation sets
train_samples, validation_samples = train_test_split(samples, test_size=0.3)
#we need to take into account the dataset augmentation
train_sample_len = len(train_samples) * (augment_factor + 1)
validation_samples_len = len(validation_samples)

print("We will train with {} samples.".format(len(train_samples)))
print("We will validate with {} samples.".format(len(validation_samples)))

#Define the generators to feed data
train_generator = generator(train_samples, batch_size= batch_size, data_augment=True, augment_factor=augment_factor)
validation_generator = generator(validation_samples, batch_size= batch_size)

#Model compilation
model = NvidiaArchitecture()
model.compile(loss='mse', optimizer='adam')
model.summary()

#Early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience= early_stopping_patience, verbose= 0, mode= 'auto')

history = model.fit_generator(train_generator, steps_per_epoch= train_sample_len / batch_size, validation_data=validation_generator, validation_steps= validation_samples_len / batch_size, epochs= max_epochs, callbacks=[early_stopping])

model.save('model.h5')



