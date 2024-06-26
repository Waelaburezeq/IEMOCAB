#%run -i libraries.py
from __future__ import print_function
#storage libraries
import boto3 as boto3
import os as os
import pickle as pickle
from google.colab import userdata
#processing libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
import inspect
from operator import itemgetter
from tqdm import tqdm  #tqdm displaty progress bar https://www.geeksforgeeks.org/python-how-to-make-a-terminal-progress-bar-using-tqdm/
from varname import nameof
import multiprocessing as mp
from torchviz import make_dot
import patoolib
from IPython.display import display, Javascript
import warnings
warnings.filterwarnings('ignore')
#text libraries
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from gensim.models import word2vec
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
#audio libraries
from scipy.io import wavfile
import librosa.display
#import scipy.io
#Model libraries
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Embedding, Dropout, LSTM,Input, Bidirectional, GlobalMaxPooling1D
from tensorflow.keras.optimizers.legacy import Adam #decay was depreciated from recent adam optimizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from keras.utils import plot_model
from keras.regularizers import l2
from keras.callbacks import LearningRateScheduler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import keras.backend as K
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import IPython.display as ipd
import seaborn as sns
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, classification_report
from torch import nn, optim
from torchvision import transforms, models, datasets
from helper_prabowo_ml import clean_html, remove_links, non_ascii, lower, email_address, removeStopWords, punct, remove_
