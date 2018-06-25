import csv
import cv2
import numpy as np
from create_model import *
from functions import *

#two generators for training and validation
train_gen = generate_next_batch()
validation_gen = generate_next_batch()

model = create_model()
history = model.fit_generator(train_gen,
                              samples_per_epoch=number_of_samples_per_epoch,
                              nb_epoch=number_of_epochs,
                              validation_data=validation_gen,
                              nb_val_samples=number_of_validation_samples,
                              verbose=1)

model.save('model.h5')
