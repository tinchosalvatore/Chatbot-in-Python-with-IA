import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import sgd_experimental

lemmatizer = WordNetLemmatizer()

intents = json.loads(open('intents.json').read())

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

## This are all the imports that we will need to make the traigning work correctly