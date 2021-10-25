import argparse

import logging, os
logging.disable(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Disable warning messages

import trainModel
import useModel


parser = argparse.ArgumentParser(description='Script for object recognition. You can train a new model or use an existing one.\nLibrary needed: opencv, tensorflow, scikit-learn and numpy', formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-t', '--train', help='Train a new model. You must write the list of objects name to detect.\nE.g: python objRecognition -t dog cat pen', nargs='+', default=None)
parser.add_argument('-n', '--nimages', help='Images number for each object to train the model. \nDefault: 100', default=100)

parser.add_argument('-m', '--model', help='Model file name to use for the recognition.\nE.g: python objRecognition -m objectGroup1', type=str, default=None)

args = parser.parse_args()

if(args.train == None and args.model == None):
    print("You must use one of -t or -m options to run the script")
    exit(0)
elif(args.train != None and args.model != None):
    print("You must use only one of -t or -m options to run the script")
    exit(0)
elif(args.train != None):
    trainModel.collect_data(args.nimages, args.train)
else:
    useModel.use_model(args.model)