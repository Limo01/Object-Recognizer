import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable warning messages

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pickle

# -----------------------------------------------------------------------------------------
# Function that uses the model given as parameter and recognize the objects
#
# model_name => model name to use
# -----------------------------------------------------------------------------------------
def use_model(model_name, print_classes):
    if(not os.path.isdir(model_name)):
        print('The model doesn\'t exists')
        return

    # Load the model
    model= load_model(model_name+'/'+model_name+'.h5')
    
    # Load the object names that the model recognize from file
    with open(model_name+'/classes.ob', 'rb') as fp:
        classes = pickle.load(fp)
    
    if print_classes:
        print('The application can recognize the following objects:')
        for c in classes:
            print(f"* {c}")

    camera = cv2.VideoCapture(0,cv2.CAP_DSHOW)
    box_size = 224
    width = int(camera.get(3))
    height= int(camera.get(4))

    while True:
        
        ret, frame = camera.read()
        
        # Break the loop if there is trouble reading the frame
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
            
        cv2.rectangle(frame, (width//2 - box_size//2, height//2 - box_size//2), (width//2 + box_size//2, height//2 + box_size//2), (250, 0, 250), 2)
    
        cv2.namedWindow("Object Recognizer", cv2.WINDOW_NORMAL)

        roi = frame[height//2 - box_size//2 : height//2 + box_size//2, width//2 - box_size//2 : width//2 + box_size//2]
        
        # Normalize the image like we did in the preprocessing step, also convert float64 array.
        roi = np.array([roi]).astype('float64') / 255.0
    
        # Get model's prediction.
        pred = model.predict(roi)
        
        # Get the index of the target class.
        target_index = np.argmax(pred[0])

        # Get the probability of the target class
        prob = np.max(pred[0])

        # Show results
        cv2.putText(frame, "Prediction: {} {:.2f}%".format(classes[target_index], prob*100 ),
                    (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 0, 255), 2, cv2.LINE_AA)
                
        cv2.putText(frame, 'Press \'q\' to quit the program', (10, height-30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        
        cv2.imshow("Object Recognizer", frame)  
    
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    camera.release()
    cv2.destroyAllWindows()