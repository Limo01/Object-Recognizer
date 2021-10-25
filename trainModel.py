import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable warning messages

import pickle
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense,Dropout,GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------------------
# Function that retruns a dictionary of empty lists where the keys are the classs names
#
# classes => list of strings that are the object names to detect
# -----------------------------------------------------------------------------------------
def __init_data(classes):
    data= dict()

    for c in classes:
        data[c] = []

    return data

# -----------------------------------------------------------------------------------------
# Function to train a new model
#
# classes => list of strings that are the object names to detect
# data => dictionary of empty lists where the keys are the object names
# -----------------------------------------------------------------------------------------
def __train(classes, data):
    # Combine the labels of all classes together
    labels=[]
    for c in classes:
        labels= labels + [c for _ in data[c]]
    
    # Combine the images of all classes together
    images=[]
    for c in classes:
        images= images + data[c]
    
    # Normalize the images by dividing by 255, now our images are in range 0-1. This will help in training.
    images = np.array(images, dtype="float") / 255.0
    
    # Print out the total number of labels and images.
    print('Total images: {} , Total Labels: {}'.format(len(labels), len(images)))

    # Create an encoder Object
    encoder = LabelEncoder()

    # Convert Lablels to integers. i.e. first_obj = 0, second_obj = 1, third_obj = 2 (mapping is done in alphabatical order)
    Int_labels = encoder.fit_transform(labels)

    # Now the convert the integer labels into one hot format. i.e. 0 = [1,0,0,0]  etc.
    one_hot_labels = to_categorical(Int_labels, 4)

    # Now we're splitting the data, 75% for training and 25% for testing.
    (trainX, testX, trainY, testY) = train_test_split(images, one_hot_labels, test_size=0.25, random_state=50)

    # Empty memory from RAM
    images = []
    data= []

    # This is the input size which our model accepts.
    image_size = 224

    # Loading pre-trained NASNETMobile Model without the head by doing include_top = False
    N_mobile = tf.keras.applications.NASNetMobile( input_shape=(image_size, image_size, 3), include_top=False, weights='imagenet')

    # Freeze the whole model 
    N_mobile.trainable = False
        
    # Adding our own custom head
    # Start by taking the output feature maps from NASNETMobile
    x = N_mobile.output

    # Convert to a single-dimensional vector by Global Average Pooling. 
    # We could also use Flatten()(x) GAP is more effective reduces params and controls overfitting.
    x = GlobalAveragePooling2D()(x)

    # Adding a dense layer with 712 units
    x = Dense(712, activation='relu')(x) 

    # Dropout 40% of the activations, helps reduces overfitting
    x = Dropout(0.40)(x)

    # The fianl layer will contain 4 output units (no of units = no of classes) with softmax function.
    preds = Dense(4,activation='softmax')(x) 

    # Construct the full model
    model = Model(inputs=N_mobile.input, outputs=preds)

    # Check the number of layers in the final Model
    print ("Number of Layers in Model: {}".format(len(model.layers[:])))

    augment = ImageDataGenerator(   
            rotation_range=30,
            zoom_range=0.25,
            width_shift_range=0.10,
            height_shift_range=0.10,
            shear_range=0.10,
            horizontal_flip=False,
            fill_mode="nearest"
    )

    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Set batchsize according to your system
    epochs = 15
    batchsize = 20

    # Start training
    model.fit(x=augment.flow(trainX, trainY, batch_size=batchsize), validation_data=(testX, testY), steps_per_epoch= len(trainX) // batchsize, epochs=epochs)

    name= input('Write a file name for the model (Will be created a folder with the same name): ')

    while(os.path.isdir(name)):
        name= input('Folder '+name+' already exists. Please choice another name for the model: ')

    os.mkdir(name)
    model.save(name+'/'+name+'.h5')
    
    with open(name+'/classes.ob', 'wb') as fp:
        pickle.dump(classes, fp) 


# -----------------------------------------------------------------------------------------
# Function to collect all the image to create the model
# 
# n => int, images number for each object to train the model
# classes => list of strings that are the object names to detect
# -----------------------------------------------------------------------------------------
def collect_data(n, classes):
    # Add the class 'nothing': no object detected
    classes.append('nothing')
    classes.sort()
    
    data= __init_data(classes)

    # Init the camera
    camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    
    # Tells us when to start recording
    recording = False

    # Counter keeps count of the number of samples collected
    counter = 0

    # This the ROI size
    box_size = 224

    # Getting the width of the frame from the camera properties
    width = int(camera.get(3))

    # Getting the height of the frame from the camera properties
    height= int(camera.get(4))

    # Index of the class for which collect the images
    c = 0

    while c<len(classes):
        # Read frame by frame
        ret, frame = camera.read()

        # Flip the frame laterally
        frame = cv2.flip(frame, 1)

        # Exit if there is trouble reading the frame
        if not ret:
            exit(0)

        # Define ROI for capturing samples
        cv2.rectangle(frame, (width//2 - box_size//2, height//2 - box_size//2), (width//2 + box_size//2, height//2 + box_size//2), (250, 0, 250), 2)

        # Make a resizable window.
        cv2.namedWindow("Train model", cv2.WINDOW_NORMAL)        

        if recording:
            # Grab only slected roi
            roi = frame[height//2 - box_size//2 : height//2 + box_size//2, width//2 - box_size//2 : width//2 + box_size//2]
            
            # Append the roi and class name to the list with the selected class_name
            data[classes[c]].append(roi)
                                    
            # Increment the counter 
            counter += 1 
        
            # Text for the counter
            text = "Collected Samples of {}: {}".format(classes[c], counter)         

            if(counter==n):
                c += 1
                counter = 0
                recording = False
        else:
            text = "Press \'r\' to start colleting images of {}".format(classes[c])

        cv2.putText(frame, text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Press \'q\' to quit the program', (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1, cv2.LINE_AA)

        cv2.imshow("Train model", frame)

        k = cv2.waitKey(1)
        
        if(k==ord('r')):
            recording= True
        elif(k==ord('q')):
            exit(0)

    camera.release()
    cv2.destroyAllWindows()

    __train(classes, data)