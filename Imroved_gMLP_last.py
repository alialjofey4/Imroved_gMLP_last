from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Embedding
from keras.layers import SpatialDropout1D
from tensorflow import keras
#from tensorflow.keras.layers import LayerNormalization, MultiHeadAttention, Dropout, Add, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
#from keras.layers.merge import concatenate
from keras.layers import concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, MaxPooling2D,Concatenate
import seaborn as sns
import requests
import psycopg2
import pandas as pd
import numpy as np
from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
#from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional
from tcn import TCN, tcn_full_summary
from matplotlib import pyplot
import matplotlib
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.layers import GRU
from sklearn.feature_extraction.text import TfidfVectorizer
from keras import regularizers
from keras.preprocessing import sequence
from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import LSTM, Reshape, Bidirectional
from tensorflow import math, matmul, reshape, shape, transpose, cast, float32 
from keras.layers import Activation, BatchNormalization
from keras.layers import AveragePooling1D, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import (EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout, GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense, TimeDistributed)
#from keras_self_attention import SeqSelfAttention
from tensorflow.keras.utils import to_categorical, custom_object_scope
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm, tqdm_pandas
import scipy
from scipy.stats import skew
import json
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import seaborn as sns
import glob 
import os
import sys
import pickle
import warnings
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from keras.callbacks import EarlyStopping
#import tensorflow_hub as hub
from keras.utils.np_utils import to_categorical
import itertools
import csv
import torch
#import tokenization
#from bert import tokenization
import os
import re
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Dense, Layer 
from transformers import GPT2Tokenizer
from transformers import AutoModel, AutoTokenizer 
from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report

from tensorflow.keras import backend as K

from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Flatten


import sys
from absl import flags
sys.argv=['preserve_unused_tokens=False']
flags.FLAGS(sys.argv)
with tf.device('/gpu:0'):
        matplotlib.rcParams['font.family'] = 'Times New Roman'
        matplotlib.rcParams['font.size'] = 14
        sns.set(font_scale=1.4)

        good_plain,good_noicy, good_hy, good_urls = [], [],[],[]
        bad_plain, bad_noicy, bad_hy, bad_urls = [], [],[],[]
        labels, text=[],[]
        train_u, train_h=[],[]
        test_u, test_h=[],[]
        texts, labels=[],[]
        
        
        maxlen=200
        
        
##
#        with open("/home/ali/anaconda3/dataset1.txt", 'r',encoding='latin-1') as file:
#              for line in file:
#                  columns = line.strip().split('|')
#                  #columns = line.strip().split('\t')
#                  texts.append(columns[0])
#                  if (columns[1]=='+1'):
#                      labels.append('1')
#                  else:
#                      labels.append('0')

#
        import pandas as pd
        
        #new reading 2
#
#        # Define paths
#        file_path ="/home/ali/anaconda3/PhiUSIIL_Phishing_URL_Dataset.csv"  # Update to the actual CSV file path
#        # Read the CSV file
#        df = pd.read_csv(file_path)
#        # Check the structure of the CSV
#        print(f"Number of rows: {df.shape[0]}")
#        print(df.head())  # Check the first few rows to ensure proper formatting
#        
#        if 'URL' in df.columns and 'Label' in df.columns:
#           # Process the CSV rows
#           for index, row in df.iterrows():
#               texts.append(row['URL'])
#       
#           # Convert 'bad' -> 1, 'good' -> 0
#               label = row['Label'].strip().lower()
#               if label == 'bad':
#                  labels.append(1)
#               elif label == 'good':
#                  labels.append(0)
#               else:
#                  print(f"Invalid label at row {index}: {label}")
#        else:
#              print("Error: CSV file does not contain 'URL' or 'Label' columns.")
              
              

         #new reading 3     
         # Define paths
#        file_path ="/home/ali/anaconda3/PhiUSIIL_Phishing_URL_Dataset.csv"  # Update to the actual CSV file path
#        # Read the CSV file
#        df = pd.read_csv(file_path)
#        # Check the structure of the CSV
#        print(f"Number of rows: {df.shape[0]}")
#        print(df.head())  # Check the first few rows to ensure proper formatting
#        
#        if 'URL' in df.columns and 'label' in df.columns:
#           # Process the CSV rows
#           for index, row in df.iterrows():
#               texts.append(row['URL'])
#       
#           # Convert 'bad' -> 1, 'good' -> 0
#               label = row['label']
#               if label == 1:
#                  labels.append(1)
#               elif label == 0:
#                  labels.append(0)
#               else:
#                  print(f"Invalid label at row {index}: {label}")
#        else:
#              print("Error: CSV file does not contain 'URL' or 'Label' columns.")     
#              
#
#        # Print the size of the lists
#        print(f"Number of texts: {len(texts)}")
#        print(f"Number of labels: {len(labels)}")
#
## The 'texts' now contains all the URLs, and 'labels' contains 1 for 'bad' and 0 for 'good'
#        trainDF = pd.DataFrame()
#        trainDF['text'] = texts
#        trainDF['label'] =labels
#        X_train, X_test,y_train, y_test = model_selection.train_test_split(trainDF['text'], trainDF['label'],test_size=0.2, random_state=0)
#        
#        #X_test=X_test[:1000]
#        #y_train=y_train[:1000]
#        #X_train=X_train[:1000]
#        #y_test=y_test[:1000]
#        encoder = preprocessing.LabelEncoder()
#        y_train = encoder.fit_transform(y_train)
#        y_test = encoder.fit_transform(y_test)
#        y_train1 = to_categorical(y_train)
#        y_test1 = to_categorical(y_test)



        

        # Custom loss function
        def custom_loss_function(y_true, y_pred):
            # Example: Weighted binary cross-entropy (can be adjusted based on the task)
            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)(y_true, y_pred)
            # Example of adding a regularization term
            reg_term = tf.reduce_mean(tf.abs(y_pred))
            return loss + 0.01 * reg_term  # 0.01 is a regularization coefficient
            
      
        def focal_loss(gamma=1.0, alpha=0.12):
             def focal_loss_fixed(y_true, y_pred):
                 # Cast y_true to float32 to match y_pred
                 y_true = tf.cast(y_true, tf.float32)
                 
                 # Clip predictions to avoid log(0)
                 y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
                 
                 # Binary cross entropy
                 bce = K.binary_crossentropy(y_true, y_pred)
                 
                 # Modulating factor
                 pt = tf.exp(-bce)
                 focal_term = (1 - pt) ** gamma
                 
                 # Final focal loss
                 loss = alpha * focal_term * bce
                 return loss
             return focal_loss_fixed


        def plot_confusion_matrix(cm,classes,normalize=False,title='Confusion matrix',cmap=plt.cm.Blues):
            pyplot.rcParams["font.family"] = "Times New Roman"
            pyplot.rcParams['font.size'] = 14
            sns.set(font_scale=1.4)
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks =np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=0)
            plt.yticks(tick_marks, classes)
            if normalize:
                    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                #print("Normalized confusion matrix")
            else:
                    1#print('Confusion matrix, without normalization')

                    #print(cm)

            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                  plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
            plt.tight_layout()
            plt.ylabel('True label')
            plt.xlabel('Predicted label')

        def bert_encode(texts, tokenizer, max_len):
            all_tokens = []
            all_masks = []
            all_segments = []
            for text in texts:
                text = tokenizer.tokenize(text)
                
                text = text[:max_len-2]
                input_sequence = ["[CLS]"] + text + ["[SEP]"]
                pad_len = max_len-len(input_sequence)
                
                tokens = tokenizer.convert_tokens_to_ids(input_sequence) + [0] * pad_len
                pad_masks = [1] * len(input_sequence) + [0] * pad_len
                segment_ids = [0] * max_len
                
                all_tokens.append(tokens)
                all_masks.append(pad_masks)
                all_segments.append(segment_ids)
                
            return np.array(all_tokens)
        #np.array(all_masks), np.array(all_segments)

        #BERT word embeding features
#        m_url = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/2'
#        #m_url2='https://tfhub.dev/google/universal-sentence-encoder/4'
#        bert_layer = hub.KerasLayer(m_url, trainable=True)
#        vocab_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
#        do_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
#        tokenizer = tokenization.FullTokenizer(vocab_file, do_lower_case)
       
#        embedding_matrix = bert_layer.get_weights()[0]
#        print("embedding_matrix:", embedding_matrix.shape)
#        train_input = bert_encode(X_train, tokenizer, max_len=max_len)
#        test_input = bert_encode(X_test, tokenizer, max_len=max_len)

        import transformers
        from transformers import TFDistilBertModel
        from transformers import DistilBertTokenizer
        from transformers import DistilBertConfig
        from transformers import TFAutoModel, AutoTokenizer
        from transformers import GPT2Model, GPT2Tokenizer
        from transformers import TFXLNetModel, XLNetTokenizer
        from transformers import AlbertTokenizer, TFAlbertModel
        from transformers import RobertaTokenizerFast
        from transformers import TFRobertaModel
        
        



        #tokenizer_gpt = AutoTokenizer.from_pretrained("/home/ali/anaconda3/gpt2")
        #tokenizer_gpt = GPT2Tokenizer.from_pretrained('gpt2')
        model_name = "/home/ali/anaconda3/gpt2/"
        tokenizer_gpt = GPT2Tokenizer.from_pretrained(model_name)
        model_gpt = GPT2Model.from_pretrained(model_name)
        tokenizer_DistilBERT = transformers.DistilBertTokenizer.from_pretrained('bert-base-uncased')
        transformer_layer = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased')
         #this for Albert model
        albert_tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
        albert_model = TFAlbertModel.from_pretrained('albert-base-v2')
       # tokenizer_roberta = RobertaTokenizerFast.from_pretrained("bert-base-uncased")
        roberta_model = TFRobertaModel.from_pretrained('bert-base-uncased')
        
        from transformers import AutoTokenizer, TFAutoModel
        
        tokenizer2 = AutoTokenizer.from_pretrained("bert-base-uncased")
        hf_model = TFAutoModel.from_pretrained("bert-base-uncased") 

        
        
         #this for xlnet model
        xlnet_model = 'xlnet-base-cased'
        xlnet_tokenizer = XLNetTokenizer.from_pretrained(xlnet_model)
#        
#        train_input = bert_encode(X_train, tokenizer_gpt, max_len=maxlen)
#        test_input = bert_encode(X_test, tokenizer_gpt, max_len=maxlen)
        
        #train_input = bert_encode(X_train, xlnet_tokenizer, max_len=200)
        #test_input = bert_encode(X_test, xlnet_tokenizer, max_len=200)
        
        #train_input = bert_encode(X_train, albert_tokenizer, max_len=200)
        #test_input = bert_encode(X_test, albert_tokenizer, max_len=200)
        
        #train_input = bert_encode(X_train, tokenizer_roberta, max_len=200)
        #test_input = bert_encode(X_test, tokenizer_roberta, max_len=200)
                 
       
        #embedding_matrix = transformer_layer.weights[0].numpy()
        embedding_matrix =  model_gpt.get_input_embeddings().weight.detach().numpy() 

        
        print("embedding_matrix",embedding_matrix.shape)
        

        
              
              #this function is used to get weights matrix when we use character embeding fearures of URLs used by Zhang et al. [22] method  
        def get_embedding_weights2(text):
        
             # Initialize the tokenizer for character-level embedding
            tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
            tk.fit_on_texts(texts)
      
            # Define character and custom token indices
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
            special_tokens = ["account", "admin", "administrator", "auth", "bank", "client", "confirm", "cmd", "email", 
                        "host", "login", "password", "pay", "private", "registed", "safe", "secure", "security", 
                        "sign", "service", "signin", "submit", "user", "update", "validation", "verification", "webscr"]
      
            char_dict = {char: i + 1 for i, char in enumerate(alphabet)}
      
            # Set up indices for <PAD> and <UNK>
            pad_index = len(char_dict) + len(special_tokens) + 1
            unk_index = pad_index + 1
      
            # Update tokenizer's word index to include custom characters and tokens
            tk.word_index = char_dict.copy()
            tk.word_index.update({token: idx + len(char_dict) for idx, token in enumerate(special_tokens)})
            tk.word_index[tk.oov_token] = unk_index
            tk.word_index['<PAD>'] = pad_index
            vocab_size = len(tk.word_index) + 1  # +1 for 0 index padding
            embedding_weights2 = []
            embedding_weights2 = np.zeros((vocab_size, vocab_size))
      
            for char, idx in tk.word_index.items():
                if idx < vocab_size:
                    embedding_weights2[idx, idx] = 1.0
      
            embedding_weights2 = np.array(embedding_weights2)
            print("Embedding weights2 shape:", embedding_weights2.shape)
            print("Vocab size2:", vocab_size)
            return embedding_weights2
            
              
  
        #this function is used to get weights matrix when we use character embeding fearures of URLs used by Aljofey et al. [8] method  
        def get_embedding_weights(text):
                  #character embeding fearures
             tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
             tk.fit_on_texts(text)
             ##
             alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
             ##
             char_dict = {}
             for i, char in enumerate(alphabet):
                 char_dict[char] = i + 1
             
                 tk.word_index = char_dict.copy()
             # Add 'UNK' to the vocabulary
             tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
             vocab_size = len(tk.word_index)+1
                           # Embedding weights
             embedding_weights = []  # (70, 69)
             embedding_weights.append(np.zeros(vocab_size))  # (0, 69)
           
             for char, i in tk.word_index.items():  # from index 1 to 69
               onehot = np.zeros(vocab_size)
               onehot[i - 1] = 1
               embedding_weights.append(onehot)
           
             embedding_weights = np.array(embedding_weights)
             return embedding_weights
              
             
           

        class gMLPLayer(layers.Layer):
                def __init__(self, num_patches, embedding_dim, dropout_rate, *args, **kwargs):
                    super(gMLPLayer, self).__init__(*args, **kwargs)
                    self.num_patches = num_patches
                    self.embedding_dim=embedding_dim
                    self.dropout_rate=dropout_rate
                
                    
                     #Define CNN Projections to replace TCN layers and add GlobalMaxPooling1D
                    self.cnn_projection1 = Sequential([
                        layers.Conv1D(512, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.Conv1D(512, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.Conv1D(512, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.Conv1D(512, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.SpatialDropout1D(0.1),
                        layers.GlobalMaxPooling1D()  # Added GlobalMaxPooling1D
                    ])
                    
                    self.cnn_projection2 = Sequential([
                        layers.Conv1D(128, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.Conv1D(128, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.Conv1D(128, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.Conv1D(128, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.SpatialDropout1D(0.1),
                        layers.GlobalMaxPooling1D()  # Added GlobalMaxPooling1D
                    ])
                    
                    self.cnn_projection3 = Sequential([
                        layers.Conv1D(256, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.Conv1D(256, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.Conv1D(256, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.Conv1D(256, kernel_size=3, padding='causal', dilation_rate=1, activation='relu'),
                        layers.SpatialDropout1D(0.1),
                        layers.GlobalMaxPooling1D()  # Added GlobalMaxPooling1D
                    ])
                    


                    self.lstm_projection = Sequential([layers.Bidirectional(layers.LSTM(embedding_dim, return_sequences=True)),])
                    
                    
                    self.channel_projection1 = keras.Sequential([layers.Dense(units=3712),
                              layers.ReLU(),
                              layers.Dropout(rate=dropout_rate),
                                                            
                          ])
                      
                      

                    self.channel_projection2 = layers.Dense(units=3712)

                    self.spatial_projection = layers.Dense(
                          units=200, bias_initializer="Ones"
                      )

                    self.normalize1 = layers.LayerNormalization(epsilon=1e-6)
                    self.normalize2 = layers.LayerNormalization(epsilon=1e-6)
                    self.normalize3 = layers.LayerNormalization(epsilon=1e-6)

                     # Residual Connection Layer
                    self.residual_dense = layers.Dense(units=3712)

                def spatial_gating_unit(self, x):
                   u, v = tf.split(x, num_or_size_splits=2, axis=2)
                   v = self.normalize2(v)
               
                 
                   # Transpose to ensure correct shape
                   v_channels = tf.linalg.matrix_transpose(v)
                   v_projected = self.spatial_projection(v_channels)
                   v_projected = tf.linalg.matrix_transpose(v_projected)
                 
                   return u * v_projected


                def get_config(self):
                    config = super(gMLPLayer, self).get_config()
                    config.update({
                        'num_patches': self.num_patches,
                        'embedding_dim': self.embedding_dim,
                        'dropout_rate': self.dropout_rate,
                    })
                    return config

                def call(self, inputs ):
                
                 # Use explicit TensorFlow operations instead of Python unpacking
                  token_inputs = inputs[0]
                  char_inputs = inputs[0]
                  
                 

                     # Apply CNN to extract spatial features
                  x_cnn1 = self.cnn_projection1(token_inputs)
                  x_cnn2 = self.cnn_projection2(token_inputs)
                  x_cnn3 = self.cnn_projection3(token_inputs)
                  
                  concatenated = Concatenate()([ x_cnn1, x_cnn2, x_cnn3])
                  #old
                  concatenated = tf.expand_dims(concatenated, axis=1)
                   # Tile CNN output to match LSTM sequence length
                  #old
                  concatenated = tf.tile(concatenated, [1, self.num_patches, 1])  # Repeat across num_patches
                             
                     
                     #old
                  x_lstm = self.lstm_projection(token_inputs)
                  
                  

                  #old
                  # Combine CNN and LSTM outputs with the original inputs
                  char_inputs = tf.expand_dims(char_inputs, axis=2)  # Example reshape, adjust as needed
                  char_inputs = tf.squeeze(char_inputs, axis=2)  # Removes the dimension with size 1
                  x_combined = tf.concat([concatenated, x_lstm, char_inputs], axis=2)
                
                  
                  x_combined = self.normalize1(x_combined)
                  x_projected = self.channel_projection1(x_combined)
                  x_spatial = self.spatial_gating_unit(x_projected)
                  x_projected = self.channel_projection2(x_spatial)
                  # Apply residual connection
                  # Apply residual connection
                  residual = self.residual_dense(x_combined)  # Shape: [batch_size, embedding_dim]
                
                  
                  return x_combined + x_projected + residual

        def build_classifier(blocks,embedding_weights,vocab_size=123, input_shape=maxlen,embedding_dim=1024, num_classes=1):
                        
                        inputs1 = tf.keras.layers.Input(shape = (maxlen,))
                        inputs2 = tf.keras.layers.Input(shape = (maxlen,))
                        

                        x1 = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=False)(inputs1)
                        x2 = Embedding(vocab_size+1, 124, weights=[embedding_weights])(inputs2)
                        

                        #concatenated1 = Concatenate()([cnn6,cnn7,cnn8])
                    
                        b1 = blocks[0]([x1,x2])
                        b1 = blocks[1]([x1,x2])
                        b1 = blocks[2]([x1,x2])
                        b1 = blocks[3]([x1,x2])
                        b1 = blocks[4]([x1,x2])
                    
                        
                        #concatenated1 = Concatenate()([b1,b2])
                        
                        gvp = layers.GlobalMaxPooling1D()(b1)
                                                             
                        representation = layers.Dropout(rate=dropout_rate)(gvp)
                        logits = layers.Dense(1, activation='sigmoid')(representation)
                        model=Model(inputs=[inputs1,inputs2], outputs=logits)
                        model.compile(Adam(lr=2e-5),loss=custom_loss_function,metrics=['accuracy'])
                        print(model.summary())
                        return model
                        
                        
                        
                        
                     #this function is used for character embeding fearures of URLs
        def charcter_embedding(texts, max_len):
            tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
            tk.fit_on_texts(texts)
            ##
            alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
            ##
            char_dict = {}
            for i, char in enumerate(alphabet):
                char_dict[char] = i + 1
            
                tk.word_index = char_dict.copy()
            # Add 'UNK' to the vocabulary
            tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
            
            sequences = tk.texts_to_sequences(texts)
            #test_texts = tk.texts_to_sequences(valid_x)
            
            ### Padding
            data = pad_sequences(sequences, maxlen=200, padding='post')
            print('Load')
            return data
                       
        #this extended function that used for character embeding fearures of URLs used by Zhang et al. [22] method    
        def charcter_embedding2(texts, max_len):
          
              # Initialize the tokenizer for character-level embedding
              tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
              tk.fit_on_texts(texts)
        
              # Define character and custom token indices
              alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
              special_tokens = ["account", "admin", "administrator", "auth", "bank", "client", "confirm", "cmd", "email", 
                          "host", "login", "password", "pay", "private", "registed", "safe", "secure", "security", 
                          "sign", "service", "signin", "submit", "user", "update", "validation", "verification", "webscr"]
        
              char_dict = {char: i + 1 for i, char in enumerate(alphabet)}
        
              # Set up indices for <PAD> and <UNK>
              pad_index = len(char_dict) + len(special_tokens) + 1
              unk_index = pad_index + 1
        
              # Update tokenizer's word index to include custom characters and tokens
              tk.word_index = char_dict.copy()
              tk.word_index.update({token: idx + len(char_dict) for idx, token in enumerate(special_tokens)})
              tk.word_index[tk.oov_token] = unk_index
              tk.word_index['<PAD>'] = pad_index
        
              # Convert training and test sequences to integer sequences
              sequences = tk.texts_to_sequences(texts)
             
        
              #Pad the sequences to ensure consistent input size
              #max_len = 150
              data = pad_sequences(sequences, maxlen=max_len, padding='post', value=pad_index)
              
              return data
              
              
        #this function is used to get weights matrix when we use character embeding fearures of URLs used by Zhang et al. [22] method  
        def get_embedding_weights2(text):
          
               # Initialize the tokenizer for character-level embedding
              tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK', lower=True)
              tk.fit_on_texts(texts)
        
              # Define character and custom token indices
              alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
              special_tokens = ["account", "admin", "administrator", "auth", "bank", "client", "confirm", "cmd", "email", 
                          "host", "login", "password", "pay", "private", "registed", "safe", "secure", "security", 
                          "sign", "service", "signin", "submit", "user", "update", "validation", "verification", "webscr"]
        
              char_dict = {char: i + 1 for i, char in enumerate(alphabet)}
        
              # Set up indices for <PAD> and <UNK>
              pad_index = len(char_dict) + len(special_tokens) + 1
              unk_index = pad_index + 1
        
              # Update tokenizer's word index to include custom characters and tokens
              tk.word_index = char_dict.copy()
              tk.word_index.update({token: idx + len(char_dict) for idx, token in enumerate(special_tokens)})
              tk.word_index[tk.oov_token] = unk_index
              tk.word_index['<PAD>'] = pad_index
              vocab_size = len(tk.word_index) + 1  # +1 for 0 index padding
              embedding_weights2 = []
              embedding_weights2 = np.zeros((vocab_size, vocab_size))
        
              for char, idx in tk.word_index.items():
                  if idx < vocab_size:
                      embedding_weights2[idx, idx] = 1.0
        
              embedding_weights2 = np.array(embedding_weights2)
              print("Embedding weights2 shape:", embedding_weights2.shape)
              print("Vocab size2:", vocab_size)
              return embedding_weights2
        
                   
                      #this function is used to get weights matrix when we use character embeding fearures of URLs used by Aljofey et al. [8] method  
        def get_embedding_weights(text):
                  #character embeding fearures
             tk = Tokenizer(num_words=None, char_level=True, oov_token='UNK')
             tk.fit_on_texts(text)
             ##
             alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
             ##
             char_dict = {}
             for i, char in enumerate(alphabet):
                 char_dict[char] = i + 1
             
                 tk.word_index = char_dict.copy()
             # Add 'UNK' to the vocabulary
             tk.word_index[tk.oov_token] = max(char_dict.values()) + 1
             vocab_size = len(tk.word_index)+1
                           # Embedding weights
             embedding_weights = []  # (70, 69)
             embedding_weights.append(np.zeros(vocab_size))  # (0, 69)
           
             for char, i in tk.word_index.items():  # from index 1 to 69
               onehot = np.zeros(vocab_size)
               onehot[i - 1] = 1
               embedding_weights.append(onehot)
           
             embedding_weights = np.array(embedding_weights)
             return embedding_weights
         
               
     
        #inp_train = get_inputs(train_x, xlnet_tokenizer)
        #inp_test = get_inputs(valid_x, xlnet_tokenizer)
        
              #this function is to create both of ropberta and albert models to compar..., we just need to change the parametrs...
        def create_roberta_model(bert_model, max_len=200):
            
            input_ids = Input(shape=(max_len,),dtype='int32')
            #attention_masks = Input(shape=(max_len,),dtype='int32')
            output = bert_model(input_ids)
            output = output[1]
            output = Dense(1, activation='sigmoid', name='outputs')(output)
            model = Model(inputs = [input_ids],outputs = output)
            opt = tf.keras.optimizers.Adam(learning_rate=1e-5, decay=1e-7)
            model.compile(opt, loss='binary_crossentropy', metrics=['accuracy'])
            print(model.summary()) 
            return model
              
              
          #this function is to create XLnet model that we compared with our model...
        def create_xlnet(mname):
            """ Creates the model. It is composed of the XLNet main block and then
            a classification head its added
            """
            # Define token ids as inputs
            word_inputs = tf.keras.Input(shape=(200,), name='word_inputs', dtype='int32')
        
            # Call XLNet model
            xlnet = TFXLNetModel.from_pretrained(mname)
            xlnet_encodings = xlnet(word_inputs)[0]
        
            # CLASSIFICATION HEAD 
            # Collect last step from last hidden state (CLS)
            doc_encoding = tf.squeeze(xlnet_encodings[:, -1:, :], axis=1)
            # Apply dropout for regularization
            doc_encoding = tf.keras.layers.Dropout(.1)(doc_encoding)
            # Final output 
            outputs = tf.keras.layers.Dense(1, activation='sigmoid', name='outputs')(doc_encoding)
        
            # Compile model
            model = tf.keras.Model(inputs=[word_inputs], outputs=[outputs])
            model.compile(optimizer=tf.keras.optimizers.Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
            xlnet.summary()
        
            return model
                    
              
        #Zhang_2021 et al. [23] method     
        def CNN_biLSTM(embedding_weights1,vocab_size=123):  
            inputs = tf.keras.layers.Input(shape = (200,))
            x1 = Embedding(vocab_size + 1, 124, weights=[embedding_weights1])(inputs)
            
            x1=Dropout(0.25)(x1)
            x1=Conv1D(128, 8, activation='relu')(x1)  
            x1=Conv1D(128, 10, activation='relu')(x1)
            x1=MaxPooling1D(pool_size=2)(x1)
            x1=Conv1D(256, 12, activation='relu')(x1)
            x1=MaxPooling1D(pool_size=2)(x1)
            x1=LSTM(64, return_sequences=True, recurrent_dropout=0.2)(x1)
            x1=MaxPooling1D(pool_size=2)(x1)
            x1=LSTM(128, return_sequences=True, recurrent_dropout=0.2)(x1)
            output=Dense(1024,activation='relu')(x1)
            flat = Flatten()(output)    
            outputs=Dense(1,activation='sigmoid')(flat)
            model3 = Model(inputs=inputs, outputs=outputs)
            #model3.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
            model3.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
            print(model3.summary())
            return model3
            
            
             # Wei_2020 et al. [24] method
        def wei_wei (embedding_weights1,vocab_size=95):
                     inputs2 = tf.keras.layers.Input(shape=(200,))
                     x4 = Embedding(vocab_size + 1, 96, weights=[embedding_weights1])(inputs2)
                     x4 = Conv1D(64, kernel_size=8, activation='relu')(x4)
                     x4 = MaxPooling1D(pool_size=2)(x4)
       
                     x4 = Conv1D(16, kernel_size=16, activation='relu')(x4)
                     x4 = MaxPooling1D(pool_size=2)(x4)
       
                     x4 = Conv1D(8, kernel_size=32, activation='relu')(x4)
                     x4 = MaxPooling1D(pool_size=2)(x4)
       
                     x4 = Flatten()(x4)  # Add this line to flatten the output
       
                     out = Dense(32, activation='relu')(x4)
                     out = Dense(1, activation='sigmoid')(out)
       
                     model = Model(inputs=inputs2, outputs=out)
                     model.compile(Adam(lr=2e-5), loss='binary_crossentropy', metrics=['accuracy'])
                     print(model.summary())
                     return model
                     
                     
        def TCN_1(embedding_matrix,length=200,kernel_size = 3, activation='relu'):
                     inp = Input( shape=(length,))
                     x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=True)(inp)
                     x = SpatialDropout1D(0.1)(x)
                     dilations = [1, 2, 4, 8, 16]
                 
                 # Define the TCN model using Conv1D layers
                     for dilation_rate in dilations:
                         x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, dilation_rate=dilation_rate, activation='relu', padding='causal')(x)
                 ##
                 
                 ##    x = TCN(128,dilations = [1, 2, 4], return_sequences=True, activation = activation, name = 'tcn1')(x)
                 ##    x = TCN(64,dilations = [1, 2, 4], return_sequences=True, activation = activation, name = 'tcn2')(x)
                     avg_pool = GlobalAveragePooling1D()(x)
                     max_pool = GlobalMaxPooling1D()(x)
                     conc = concatenate([avg_pool, max_pool])
                     conc = Dense(16, activation="relu")(conc)
                     conc = Dropout(0.1)(conc)
                     outp = Dense(1, activation="sigmoid")(conc)
                     model = Model(inputs=inp, outputs=outp)
                     model.compile( loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
                     print(model.summary()) 
                     return model
                                    
       
            
            
                    
                    # Alshehri et al. [23] method
        def Mohammed_2022 (embedding_weights1,vocab_size=95):
                      inputs2 = tf.keras.layers.Input(shape = (200,))
                      x4 = Embedding(vocab_size + 1, 96, weights=[embedding_weights1])(inputs2)
                      c1=Conv1D (128, kernel_size=4, activation='relu')(x4)
                      c2=Conv1D (128, kernel_size=6, activation='relu')(x4)
                      c3=Conv1D (128, kernel_size=10, activation='relu')(x4)
                      c4=Conv1D (128, kernel_size=20, activation='relu')(x4)
        
                      flat1 = Flatten()(c1)
                      flat2 = Flatten()(c2)
                      flat3 = Flatten()(c3)
                      flat4 = Flatten()(c4)
                      flat5 = Flatten()(x4)
        
                      concatenated1 = Concatenate()([flat1,flat2,flat3,flat4,flat5])
        
                      concatenated1=Dropout(0.5)(concatenated1)
        
                      concatenated1 = Dense(64, activation='relu')(concatenated1)
                      concatenated1 = Dense(64, activation='relu')(concatenated1)
                      concatenated1 = Dense(64, activation='relu')(concatenated1)
        
                      logits = layers.Dense(1, activation='sigmoid')(concatenated1)
                      model=Model(inputs=inputs2, outputs=logits)
                                #model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
                      model.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
                      print(model.summary())
                      return model
                      
                      
                      
                       # aljofey_2020 et al. [8] method
        def aljofey_2020 (embedding_weights1,vocab_size=95):
                         inputs = tf.keras.layers.Input(shape = (200,))
                         x = Embedding(vocab_size + 1, 96, weights=[embedding_weights1])(inputs)              
                         conv_layers = [[256, 7, 3],
                              [256, 7, 3],
                              [256, 3, -1],
                              [256, 3, -1],
                              [256, 3, -1],
                              [256, 3, -1],
                              [256, 3, 3]]
         
                         fully_connected_layers = [2028, 2048]
                         dropout_p = 0.5
                         #optimizer = 'adam'
                         for filter_num, filter_size, pooling_size in conv_layers:
                             x = Conv1D(filter_num, filter_size)(x)
                             x = Activation('relu')(x)
                             if pooling_size != -1:
                                 x = MaxPooling1D(pool_size=pooling_size)(x)  # Final shape=(None, 34, 256)
                         x = Flatten()(x)  # (None, 8704)
         
                         for dense_size in fully_connected_layers:
                           x = Dense(dense_size, activation='relu')(x)  # dense_size == 1024
                           x = Dropout(dropout_p)(x)
         # Output Layer
                         out = Dense(1,activation='sigmoid')(x)
         # Build model
                         model = Model(inputs=inputs, outputs=out)
                         model.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
                         print(model.summary())
                         return model
                         
                         
        def CNN_LSTM(embedding_weights):

            inputs = tf.keras.layers.Input(shape = (200,))
            #x1 = Embedding(89, 1024,trainable=True)(inputs)

            x1 = Embedding(*embedding_weights.shape, weights=[embedding_weights],trainable=False)(inputs)

            #x1 = Embedding(96 , 96, weights=[embedding_weights])(inputs)
            
#            x1=Dropout(0.25)(x1)
#            x1=Conv1D(128, 8, activation='relu')(x1)
#            
#            x1=Conv1D(128, 10, activation='relu')(x1)
#            x1=MaxPooling1D(pool_size=2)(x1)
#            x1=Conv1D(256, 12, activation='relu')(x1)
#            x1=MaxPooling1D(pool_size=2)(x1)
            x1=layers.Bidirectional(LSTM(1024, return_sequences=True, recurrent_dropout=0.2))(x1)
            #x1=MaxPooling1D(pool_size=2)(x1)
            #x1=layers.Bidirectional(LSTM(768, return_sequences=True, recurrent_dropout=0.2))(x1)
            output=Dense(768,activation='relu')(x1)
            flat = Flatten()(output)    
            outputs=Dense(1,activation='sigmoid')(flat)
            model3 = Model(inputs=inputs, outputs=outputs)
            #model3.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
            model3.compile(Adam(lr=2e-5),loss='binary_crossentropy',metrics=['accuracy'])
            print(model3.summary())
            return model3
            
            
        
        def DNN_model2(embedding_weights):
                  inputs = tf.keras.layers.Input(shape = (200,))
                  #x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=False)(inputs)

                  #inputs = layers.Input(shape=89,)
                  #x = Embedding(89, 1024,trainable=True)(inputs)

                  x = Embedding(96, 96, weights=[embedding_weights])(inputs)
                  x = layers.Dense(64, activation='relu')(x)
                   ### hidden layer 2
                  x = layers.Dense(32, activation='relu')(x)
                  x=Dropout(0.1)(x)
                  x=layers.Dense(16, activation='relu')(x)
                  x=layers.Dense(8, activation='relu')(x)
                  out = Dense(1,activation='sigmoid')(x)
                  model = Model(inputs=[inputs], outputs=out)
                  model.compile(optimizer = 'adam',loss='binary_crossentropy',metrics=['accuracy'])
                  print(model.summary())
                  return model
                         
                         
        def build_rnn_model(embedding_weights):
                    # Input layer for sequences of length 200
                    inputs = tf.keras.layers.Input(shape=(200,))
                    x = Embedding(96, 96, weights=[embedding_weights])(inputs)
                    #inputs = tf.keras.layers.Input(shape = (89,))
                       # Embedding layer Initialization
                      #x1 = Embedding(*embedding_matrix.shape, weights=[embedding_matrix],trainable=False)(inputs1)
                    #x1 = Embedding(60, 1024,trainable=True)(inputs)
                    # Embedding layer using pretrained embeddings
                    #x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(inputs)
                    # First RNN layer, returning sequences (output shape: (batch_size, timesteps, 64))
                    x = SimpleRNN(64, return_sequences=True)(x)
                    #x = LSTM(64, return_sequences=True)(x)
##                    x=Conv1D(64, 3, activation='relu')(x)
##                    x=Conv1D(128, 5, activation='relu')(x)
##                    x=Conv1D(128, 10, activation='relu')(x)
                    # Second RNN layer, returning sequences (output shape: (batch_size, timesteps, 64))
                    x = SimpleRNN(64, return_sequences=True)(x)
                    #x = LSTM(64, return_sequences=True)(x)
##                    # Third RNN layer, no sequences returned (output shape: (batch_size, 128))
##                    x = SimpleRNN(128, return_sequences=False)(x)
                    # Dense layer
                    x = Dense(64, activation='relu')(x)
                    # Dropout layer
                    x = Dropout(0.5)(x)
                    flat = Flatten()(x)    
                    # Output layer for binary classification
                    out = Dense(1, activation='sigmoid')(flat)
                    # Build the model
                    model = Model(inputs=[inputs], outputs=out)
                    # Compile the model
                    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
                    # Print the summary of the model
                    print(model.summary())
                    return model
                               
        
  


        num_patches = 200
        dropout_rate = 0.2
        embedding_dim = 1024  # Number of hidden units.
        num_blocks = 5 # Number of blocks.
        #gmlp_blocks = keras.Sequential ([gMLPLayer(num_patches, embedding_dim, dropout_rate) for _ in range(num_blocks)])
        gmlp_blocks = [gMLPLayer(num_patches, embedding_dim, dropout_rate),gMLPLayer(num_patches, embedding_dim, dropout_rate),gMLPLayer(num_patches, embedding_dim, dropout_rate), gMLPLayer(num_patches, embedding_dim, dropout_rate),gMLPLayer(num_patches, embedding_dim, dropout_rate), gMLPLayer(num_patches, embedding_dim, dropout_rate)] 
        
        learning_rate = 0.003

        # Create an ensemble of models
        #num_models = 3
        #ensemble_models = [build_classifier(gmlp_blocks) for _ in range(num_models)]
        
#        
#          #this for using 5-fold cross validation and spilit the datasets and save them to folders... 
#        from sklearn.model_selection import StratifiedKFold
#       
#         #Ensure the directory for saving splits exists
#        output_dir = 'Dataset6_kfold_splits/'
#        if not os.path.exists(output_dir):
#              os.makedirs(output_dir)
#              
#         #Cross-validation loop
#        #kf = KFold(n_splits=5, shuffle=True, random_state=42)
#        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
#         
#        #trainDF['text'] 
#        
#        texts = np.array(texts)
#        labels_data = np.array(labels)
#      
#        from tensorflow.keras.callbacks import EarlyStopping
#        early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
#      
#        fold_no = 1
#        for train_index, val_index in skf.split( texts , labels_data):
#            print(f'Fold {fold_no}')
#            # Split data into training and validation sets
#            
#            X_train_input_ids, X_val_input_ids = texts[train_index], texts[val_index]
#            y_train, y_val = labels_data[train_index], labels_data[val_index]
#            
#            # Save training data to CSV
#            train_df = pd.DataFrame({'X_train_input_ids': X_train_input_ids, 'y_train': y_train})
#            train_file = os.path.join(output_dir, f'train_fold_{fold_no}.csv')
#            train_df.to_csv(train_file, index=False)
#      
#            # Save validation data to CSV
#            val_df = pd.DataFrame({'X_val_input_ids': X_val_input_ids, 'y_val': y_val})
#            val_file = os.path.join(output_dir, f'val_fold_{fold_no}.csv')
#            val_df.to_csv(val_file, index=False)
#            
            #fold_no += 1
            
            

        def model_report_keras(model, loss="mse", optimizer="adam", batch_size=32, epochs=1):
                  report = {}
      
                  # Params
                  total_params = model.count_params()
                  trainable_params = np.sum([K.count_params(w) for w in model.trainable_weights])
                  non_trainable_params = total_params - trainable_params
                  report["Total Params"] = int(total_params)
                  report["Trainable Params"] = int(trainable_params)
                  report["Non-trainable Params"] = int(non_trainable_params)
      
#                  # FLOPs (analytical, per forward pass, batch_size=1)
#                  report["FLOPs (approx, per sample)"] = _estimate_flops(model)
      
                  # Model size
                  bytes_fp32 = total_params * 4
                  report["Model Size (MB)"] = round(bytes_fp32 / (1024 ** 2), 2)
      
                  # Memory info (GPU if available)
                  try:
                      gpus = tf.config.list_physical_devices('GPU')
                      if gpus:
                          tf.config.experimental.set_memory_growth(gpus[0], True)
                          mem = tf.config.experimental.get_memory_info('GPU:0')
                          report["GPU Memory (MB)"] = {k: round(v / (1024**2), 2) for k, v in mem.items()}
                      else:
                          report["GPU Memory (MB)"] = "No GPU"
                  except Exception:
                      report["GPU Memory (MB)"] = "Unavailable"
      
                  # Timing (tiny dummy fit)
                  try:
                      x_dummy = _dummy_like_model_inputs(model, batch_size=batch_size)
                      # Build a dummy y with the right shape
                      out_shape = model.output_shape
                      if isinstance(out_shape, list):  # multiple outputs not expected here
                          out_shape = out_shape[0]
                      y_feats = out_shape[-1] if isinstance(out_shape, tuple) else 1
                      y_dummy = np.random.randn(batch_size, y_feats).astype(np.float32)
      
                      model.compile(optimizer=optimizer, loss=loss)
                      start = time.time()
                      model.fit(x_dummy, y_dummy, batch_size=batch_size, epochs=epochs, verbose=0)
                      end = time.time()
                      report["Train Time (epochs={}, batch={})".format(epochs, batch_size)] = round(end - start, 4)
                  except Exception as e:
                      report["Train Time"] = f"Failed: {e}"
      
                  return report
        
        
              
        def training():
             fold_no = 3
             print(f'Running ds2 method... Fold {fold_no}')
           
             # Load training data , we can change only the name of dataset path and fold_no to load it....
             train_file = f'Dataset1_kfold_splits/train_fold_{fold_no}.csv'
             train_df = pd.read_csv(train_file)
             X_train_input_ids = train_df['X_train_input_ids'].values
             y_train = train_df['y_train'].values
             
             #X_train_input_ids= X_train_input_ids[:1000]
             #y_train=y_train[:1000]
       
             # Load validation data, we can change only the name of dataset path adn fold_no to load it....
             val_file = f'Dataset1_kfold_splits/val_fold_{fold_no}.csv'
             val_df = pd.read_csv(val_file)
             X_val_input_ids = val_df['X_val_input_ids'].values
             y_test = val_df['y_val'].values
             
             #X_val_input_ids=X_val_input_ids[:1000]
             #y_test=y_test[:1000]
             
             
             
       #      #here were extract charcter features of URL within each fold of the dataset to avoid the leak of features...
#             train_data=charcter_embedding(X_train_input_ids,maxlen)
#             test_data=charcter_embedding(X_val_input_ids,maxlen)
#             embedding_weights1=get_embedding_weights(X_train_input_ids)
       #
          #here were extract charcter features2 of URL of zhange et al. [22] method within each fold of the dataset to avoid the leak of features...
          
             #here were extract token features of URL using xlnet_tokenizer...within each fold  
             #train_input=bert_encode(X_train_input_ids, xlnet_tokenizer, maxlen)
             #test_input=bert_encode(X_val_input_ids, xlnet_tokenizer, maxlen) 
       
       
             #here were extract token features of URL using albert_tokenizer...within each fold  
             #train_input=bert_encode(X_train_input_ids, albert_tokenizer, maxlen)
             #test_input=bert_encode(X_val_input_ids, albert_tokenizer, maxlen) 
       
       
             #here were extract token features of URL using tokenizer_roberta...within each fold  
##             train_input=bert_encode(X_train_input_ids, tokenizer2 , maxlen)
##             test_input=bert_encode(X_val_input_ids, tokenizer2 , maxlen) 
       
             #here were extract token features of URL using tokenizer_DistilBERT...within each fold  
             #train_input=bert_encode(X_train_input_ids, tokenizer_DistilBERT, max_len=maxlen)
             #test_input=bert_encode(X_val_input_ids, tokenizer_DistilBERT, max_len=maxlen)
             
             
             #for our model 
             train_input = bert_encode(X_train_input_ids, tokenizer_gpt, max_len=maxlen)
             test_input = bert_encode(X_val_input_ids, tokenizer_gpt, max_len=maxlen)
#             
             train_data=charcter_embedding2(X_train_input_ids,maxlen)
             test_data=charcter_embedding2(X_val_input_ids,maxlen)
             embedding_weights2=get_embedding_weights2(X_train_input_ids)
#             
       #      
       #       # Build and train the models
               #Function to build the TCN model
             #model=TCN_1(embedding_matrix)
             
               #Function to build the hybrid CNN-LSTM model
             #model=CNN_LSTM(embedding_matrix)
             
               #Function to build both of  Hussain_2023 et al. [25] and Multi-Scale CNN methods the same function... 
             #model=CNN_Fusion(embedding_matrix)
             
             
              #build Zhang_2021 et al. [23] method  
             #model=CNN_biLSTM(embedding_weights2)
             
             #model=build_rnn_model(embedding_weights1)
             #model=CNN_LSTM(embedding_matrix)
             
             
                #build Alshehri et al. [23] method 
             #model=Mohammed_2022(embedding_weights1)
             
             
              #Build Wei_2020 et al. [24] method
             #model=wei_wei(embedding_weights1)
             
               #Build aljofey_2020 et al. [8] method 
             #model=aljofey_2020(embedding_weights1)
             
               #Build XLNet [38] model
             #model = create_xlnet(xlnet_model)
             
              #this function to build both RoBERTa [29] and ALBERT [39] models but we have to change the parametrs etc...
             #model=create_roberta_model(albert_model)
             #model = create_roberta_model(hf_model)  

             #To build our model 
             model = build_classifier(gmlp_blocks, embedding_weights2)

             
             #model=DNN_model2(embedding_weights1)
             #model=TCN_1(embedding_matrix)
             
             from sklearn.metrics import confusion_matrix, precision_recall_curve, auc, roc_auc_score, roc_curve, recall_score, classification_report
       
              # Set up EarlyStopping callback
             early_stopping = EarlyStopping(monitor='val_accuracy', patience=60  , restore_best_weights=True)
             checkpoint = ModelCheckpoint('r.h5', monitor='val_accuracy', save_best_only=True)
       
             history=model.fit([train_input, train_data], y_train, epochs=100, batch_size=64,
                    callbacks=[checkpoint,early_stopping], validation_data=([test_input, test_data], y_test), verbose=2)
                    
                    
             rep = model_report_keras(model, loss="binary_crossentropy", optimizer="adam", batch_size=64, epochs=1)
             for k, v in rep.items():
                print(f"{k}: {v}")
             model.load_weights('r.h5')
                  
             loss, acc = model.evaluate([train_input, train_data], array(y_train), verbose=0)
             print('Train Accuracy: %f' % (acc*100))
             # evaluate model on test dataset dataset
             loss, acc = model.evaluate([test_input, test_data], array(y_test), verbose=0)
             print('Test Accuracy: %f' % (acc*100))
     
             predicted = model.predict([test_input, test_data])
             predicted = predicted.ravel()
             t = [1 if prob > 0.5 else 0 for prob in predicted]
             print(metrics.classification_report(y_test, t))
             print("\n f1_score(in %):", metrics.f1_score(y_test, t)*100)
             print("model accuracy(in %):", metrics.accuracy_score(y_test, t)*100)
             print("precision_score(in %):", metrics.precision_score(y_test,t)*100)
             print("roc_auc_score(in %):", metrics.roc_auc_score(y_test,t)*100)
             print("recall_score(in %):", metrics.recall_score(y_test,t)*100)
             
             from sklearn.metrics import matthews_corrcoef
             mcc = matthews_corrcoef(y_test, t)
             print("MCC:", mcc)
             
             acc = history.history['accuracy']
             val_acc = history.history['val_accuracy']
             loss = history.history['loss']
             val_loss = history.history['val_loss']
             fpr, tpr, thresholds = roc_curve(y_test, t)
             x = range(1, len(acc) + 1)
             print("acc list:\n")
             for i in acc:
                print(",",i)
                
             print("val_acc list:\n")
             for i in val_acc:
                print(",",i)
                
             print("loss list:\n")
                
             for i in loss:
                print(",",i)
                
             print("val loss:\n")     
           
             for i in val_loss:
                print(",",i)
                     
             print("x range list:\n")
           
             for i in x:
                print(",",i) 
                
                
             print("TPR list:\n")
       #
             for i in tpr:
                print(",",i)
       #     
             print("FPR list:\n")
             
             for i in fpr:
               print(",",i)
       #
             print("thresholds list:\n")
             for i in thresholds:
               print(",",i)
       #  
       #auc = roc_auc_score(valid_y, t)
       #print("auc:\n",auc)
       #
                
             from sklearn.metrics import confusion_matrix
             cnf_matrix_tra = confusion_matrix(y_test, t)
             print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))
             class_names = [0,1]
       
             print("cnf_matrix_tra:\n")
       
             for i in cnf_matrix_tra:
               print(",",i) 
             print("cnf_matrix_tra 2:\n") 
       
             print(cnf_matrix_tra)  
            
             
            
             cnf_matrix_tra = confusion_matrix(y_test, t)
             print("Recall metric in the train dataset: {}%".format(100*cnf_matrix_tra[1,1]/(cnf_matrix_tra[1,0]+cnf_matrix_tra[1,1])))
             class_names = [0,1]
             #pyplot.figure()
             #plot_confusion_matrix(cnf_matrix_tra , classes=class_names, title='Confusion matrix')
             #pyplot.show()
             fpr, tpr, thresholds = roc_curve(y_test, t)
             roc_auc = auc(fpr,tpr)
                 #Plot ROC
                 #plt.title('Receiver Operating Characteristic')
#             pyplot.rcParams["font.family"] = "Times New Roman"
#             pyplot.rcParams['font.size'] = 14
#             pyplot.plot(fpr, tpr, 'b',label='AUC = %0.3f'% roc_auc)
#             pyplot.legend(loc='lower right')
#             pyplot.plot([0,1],[0,1],'r--')
#             pyplot.xlim([-0.1,1.0])
#             pyplot.ylim([-0.1,1.01])
#             pyplot.ylabel('True Positive Rate')
#             pyplot.xlabel('False Positive Rate')
#             pyplot.show()
     
             conf_mat = confusion_matrix(y_true=y_test, y_pred=t)
             print('Confusion matrix:\n', conf_mat)
     
#             pyplot.rcParams["font.family"] = "Times New Roman"
#             pyplot.rcParams['font.size'] = 14
#             sns.set(font_scale=1.4)
#     
#             labels = ['Class 0', 'Class 1']
#             fig = pyplot.figure()
#             ax = fig.add_subplot(111)
#             cax = ax.matshow(conf_mat, cmap=plt.cm.Blues)
#             fig.colorbar(cax)
#             ax.set_xticklabels([''] + labels)
#             ax.set_yticklabels([''] + labels)
#             pyplot.xlabel('Predicted')
#             pyplot.ylabel('Actuals')
#             pyplot.show()
#             pyplot.rcParams["font.family"] = "Times New Roman"
#             pyplot.rcParams['font.size'] = 14
             sns.set(font_scale=1.4)
     
             font_name = 'Times New Roman'
             font_size = 14
     
     
             sns.set(style="whitegrid", font_scale=1.2)
                
     
     

        #this function that we use to extract the features, build and train the models....    
        training()

      
  
        def plot_history(history):
              acc = history.history['accuracy']
              val_acc = history.history['val_accuracy']
              loss = history.history['loss']
              val_loss = history.history['val_loss']
              x = range(1, len(acc) + 1)
              
              
              import matplotlib.pyplot as plt
              from matplotlib.font_manager import FontProperties
              
              fig, ax = plt.subplots(figsize=(8, 8))
              ax.plot(x, acc, 'b', label='Training acc',linewidth=4)
              ax.plot(x, val_acc, 'r', label='Validation acc',linewidth=4)
              
              
              # Customize labels and ticks
              ax.set_ylabel('Accuracy', fontsize=21, fontweight='bold')
              ax.set_xlabel('Epochs', fontsize=21, fontweight='bold')
              #ax.tick_params(axis='both', which='major', labelsize=18,weight='bold')
              
               # Customize legend
              legend=plt.legend(loc='lower right',fontsize=18)
              for text in legend.get_texts():
                  text.set_fontweight('bold')
              
              # Customize spines (borders)
              border_color = 'black'
              border_width = 0.8
              for spine in ax.spines.values():
                  spine.set_visible(True)
                  spine.set_color(border_color)
                  spine.set_linewidth(border_width)
                  
                  
              # Hide top and right spines (borders)
              ax.spines['top'].set_visible(False)
              ax.spines['right'].set_visible(False)
                  
              plt.yticks(fontsize=18, fontweight='bold')
              plt.xticks(fontsize=18,fontweight='bold')
              
              plt.grid(False)
              
              plt.show()
          
          
          

        #plot_history(history)



        

    









