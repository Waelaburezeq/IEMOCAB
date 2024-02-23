#from helper_iemocab import *
import boto3
import os

def func_args(obj:"FunctionName")->"This function returns all input args and thier annotation":
  return inspect.getfullargspec(load_files)

def execute_cell_by_index(index):
    display(Javascript(f'IPython.notebook.execute_cells([{index}])'))

def export_file(obj,local_path,file_type,name_of_obj=False):
  #file_name = 'train_labels_one_hot.txt'
  #file_name_text = nameof(obj)

  if os.path.isdir(local_path) == True:
    pass
  else:
    os.makedirs(local_path)

  if file_type =='list':
    with open(local_path+name_of_obj+'.txt', 'w') as fp:
      for item in obj:
        fp.write("%s," %item)
    print(f"file {name_of_obj} of type {file_type} was exported as txt")
  elif file_type =='arr':
    np.savetxt(local_path+name_of_obj+'.txt', obj, delimiter=",")
    print(f"file {name_of_obj} of type {file_type} was exported as txt")
  elif file_type =='dict':
    file_n = open(local_path+name_of_obj+'.pkl','wb')
    pickle.dump(obj,file_n)
    file_n.close()
    print(f"file {name_of_obj} of type {file_type} was exported as pkl")
  else:
    print('file type is not supported')
  return None

def upload_to_aws(local_file, bucket, path, s3_file,aws_access_key_id,aws_secret_access_key):
    from botocore.exceptions import NoCredentialsError
    service = 's3'
    s3 = boto3.client('s3', aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key)
    try:
        s3.upload_file(local_file, bucket, path+s3_file)
        msg = print ('Uploading %s to Amazon S3 bucket %s' %  (local_file, bucket+path+s3_file))
        #return True
    except FileNotFoundError:
        msg = print("The file was not found")
        return False
    except NoCredentialsError:
        msg = print("Credentials not available")
        return False
    return msg

def download_aws_s3_datasets(s3_bucket_n,file_n,parent_folder_name,aws_access_key_id,aws_secret_access_key):
  service = 's3'
  bucket_name=s3_bucket_n
  session = boto3.Session(aws_access_key_id=aws_access_key_id,aws_secret_access_key=aws_secret_access_key)
  s3 = session.resource(service)
  my_bucket = s3.Bucket(bucket_name)
  try:
      for obj in my_bucket.objects.all(): #tqdm displaty progress bar https://www.geeksforgeeks.org/python-how-to-make-a-terminal-progress-bar-using-tqdm/
        path, filename = os.path.split(obj.key)
        parent_folder =  obj.key.split('/')[0]
        print(obj)
        #if path == ''
        if filename.lower()==file_n.lower() and parent_folder.lower() == parent_folder_name.lower():
          msg = print(f'Downloading file {filename} from {path}')
          #print(obj)
          if os.path.isdir(path) == True:
            msg = print(f'Downloading file {filename} from {path}')
            my_bucket.download_file(obj.key,path+'/'+filename)
          else:
            msg = print(f'Creating path {path} & Downloading file {filename} from {path}')
            os.makedirs(path)
            my_bucket.download_file(obj.key,path+'/'+filename)
  except:
    msg ='ERROR in function input'
  return msg

def load_files(path,f_type,name_of_obj:"string_format"):
  try:
      if f_type =='txt':
          #import model keys into list
          my_file = open(path +name_of_obj+'.txt', "r")
          content = my_file.read()
          name_of_obj = content.split(",")
          my_file.close()
      elif f_type == 'arr':
          from numpy import loadtxt
          name_of_obj =  loadtxt(path + name_of_obj+'.txt', comments="#", delimiter=",", unpack=False)
      elif f_type =='dict':
        #print("Dictionaries should be of file extension = pkl")
        file = open(path+name_of_obj+'.pkl','rb')
        name_of_obj = pickle.load(file)
        file.close()
  except FileNotFoundError:
    print("The file was not found")
  return name_of_obj

''' #merged with load_files
def file_to_dict(file_name,file_type):
  if file_type =='pkl':
    file = open(file_name,'rb')
    file_name = pickle.load(file)
    file.close()
  return file_name
'''

def get_labels_from_oh_code(oh_code):
    """ Takes in one-hot encoded matrix
    Returns a list of decoded categories"""
    label_code = np.argmax(oh_code, axis=1)
    label = emotion_text_df_filtered.emotion.astype('category').cat.categories[label_code]
    return list(label)

#get_labels_from_oh_code(train_labels_one_hot[:5])


def extract_features(audio_array):
        """
        Extracts 193 chromatographic features from sound file.
        including: MFCC's, Chroma_StFt, Melspectrogram, Spectral Contrast, and Tonnetz
        NOTE: this extraction technique changes the time series nature of the data
        """
        features = []
        audio_data, sample_rate = audio_array, 22050 #librosa.load(file_name) as we already parsed the file before, 22050 is across all
        stft = np.abs(librosa.stft(audio_data))

        mfcc = np.mean(librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=40).T,axis=0)
        features.extend(mfcc) # 40 = 40

        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        features.extend(chroma) # +12 = 52

        mel = np.mean(librosa.feature.melspectrogram(audio_data, sr=sample_rate).T,axis=0)
        features.extend(mel) # +128 = 180

        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        features.extend(contrast) # +7 = 187
        #More possible features to add
        #tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X, ), sr=sample_rate).T,axis=0)
        #spec_cent = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate).T, axis=0)
        #spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sample_rate).T, axis=0)
        #rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sample_rate).T, axis=0)
        #zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data).T, axis=0)
        #features.extend(tonnetz) # 6 = 193
        #features.extend(spec_cent)
        #features.extend(spec_bw)
        #features.extend(rolloff)
        #features.extend(zcr)
        return np.array(features)


def scheduler(epoch):
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * 0.1**(epoch//10)

def callback_f(model_name):
  Callback =[
    keras.callbacks.EarlyStopping(monitor ='val_loss',patience=6,verbose = 1,restore_best_weights=True),
    keras.callbacks.ModelCheckpoint(filepath =local_path+model_name+'.hdf5', verbose = 1, save_best_only = True),
    LearningRateScheduler(scheduler)]
  return Callback

def get_f1(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

def f1(): #https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/F1Score
  return tfa.metrics.F1Score(HP.n_classes, threshold=0.5)

def accuracy_dict(history_model_name,name_of_obj):
  last_v =[]
  #for k in history_text_model_lstm_embedding.history.keys():
  #    last_v.append([k,history_text_model_lstm_embedding.history[k][-1]])
  [last_v.append([k,history_model_name.history[k][-1]]) for k in history_model_name.history.keys()]
  accuracy_dictionary[name_of_obj] = last_v
  #accuracy_dictionary.update(history_model_name_txt=last_v)
  return accuracy_dictionary

class save_spectrogram():
  def __init__(self):
    self.image_path = None
    self.sr = 22050
    #print(self.image_path)

  def save_spectrogram_f_mean(self, audio_vector,unique_id, label):
      #path =  pathImage+emotion_full_dict[label]
      #print(self.image_path)
      path =  self.image_path+str(label)
      if os.path.isdir(path) == True:
        pass
      else:
        os.makedirs(path)
      try:
        save_path = path + '/' + str(unique_id) + '.jpg'
        #print("\n",save_path)
        librosa.display.waveplot(audio_vector, sr=self.sr,color='b')
        fig1 = plt.gcf()
        plt.axis('off')
        plt.draw()
        fig1.savefig(save_path, dpi=50)
        #S = librosa.feature.melspectrogram(np.asarray(y), sr=sr, n_mels=128)
        #log_S = librosa.power_to_db(S, ref=np.max)
        #librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')
      except:
        print("no spectrogram was saved for file {}".format(unique_id))
      return None

  def save_spectrogram_f_rawdata(self, audio_vector,unique_id, label):
    path =  self.image_path+str(label)
    if os.path.isdir(path) == True:
        pass
    else:
        os.makedirs(path)
    try:
        save_path = path + '/' + str(unique_id) + '.jpg'
        S = librosa.feature.melspectrogram(audio_vector, sr=self.sr) #, n_mels=128, sr = 22050 for librosa is the default
        log_S = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(log_S, sr=self.sr)#, x_axis='time', y_axis='mel')
        plt.savefig(save_path)
    except:
        print("no spectrogram was saved for file {}".format(unique_id))
    return None

# PREPROCESS THE DATA Function - Input/text processing
def preproc(df, colname):
  df[colname] = df[colname].apply(clean_html)
  df[colname] = df[colname].apply(remove_links)
  df[colname] = df[colname].apply(func=non_ascii)
  df[colname] = df[colname].apply(func=lower)
  df[colname] = df[colname].apply(func=email_address)
  # df[colname] = df[colname].apply(func=removeStopWords)
  df[colname] = df[colname].apply(func=punct)
  df[colname] = df[colname].apply(func=remove_)
  return(df)


def plot_distribution(actual, predicted, labels):
  # Create a confusion matrix
  cm = confusion_matrix(actual, predicted, labels=labels)

  # Normalize confusion matrix to allow total of 100% distribution
  cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

  # Plot confusion matrix
  plt.figure(figsize=(10, 8))
  sns.heatmap(cm_normalized, annot=True, fmt=".0%", cmap="Blues", xticklabels=labels, yticklabels=labels)
  plt.xlabel('Predicted')
  plt.ylabel('Actual')
  plt.title('Normalized Confusion Matrix')
  plt.show()

  # Print classification report
  print(classification_report(actual, predicted, labels=labels))
