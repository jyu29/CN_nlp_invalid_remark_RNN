from utlis.data_utlis import read_yml as ym
from utlis.general_utlis import *
import time
import pandas as pd
import re
import jieba
import tensorflow as tf #2.8.0
from tensorflow import keras
from tensorflow.keras.preprocessing import sequence,text
import joblib
from tensorflow.keras.metrics import Accuracy,FalsePositives
from tensorflow.keras.layers import SeparableConv1D,MaxPooling1D,Convolution1D, MaxPool1D, Flatten,GlobalAveragePooling1D

# config
config = ym('/Users/JOE/Desktop/dev/invalid_remark/config/dev.yml')
process_data_path = config['remote_data_path']['processed']
dict_path = config['remote_data_path']['dictionary']

# model parameter
max_words = config['model_parameter']['max_words']
max_length = config['model_parameter']['max_len']
split_fraction = config['model_parameter']['data_split']
vocab_size= config['model_parameter']['vocab_size']
embedding_dim= config['model_parameter']['embeddiing_dim']
batch_size= config['model_parameter']['batch_size']
learning_rate= config['model_parameter']['learning_rate']
epochs= config['model_parameter']['epochs']

# df = pd.read_parquet(process_data_path +'invalid_remark_raw_masterdata_v3.gzip')
df = pd.read_csv(process_data_path +'invalid_remark_raw_masterdata_v2.csv')
######################function #################################################

def read_max_seq():
    with open(r'/max_seq_id.txt','r') as f:
        max_id=f.read()
#     print(max_id)
    return max_id


def load_stopword(dict_path):
#     jieba.load_userdict(r"/data/jyu29/nlp/text classfication/ilsten/data/stopwordmaster.txt")
    jieba.load_userdict(dict_path+"joe_remark_dict.txt")
    with open(dict_path +'stopword.txt',encoding='utf8') as f:

        line_list=f.readlines()
        stopword_list=[k.strip() for k in line_list]
        stopword_set=set(stopword_list)
#         print('停顿词列表，即变量stopword_list中共有%d个元素' %len(stopword_list))
#         print('停顿词集合，即变量stopword_set中共有%d个元素' %len(stopword_set))
    return stopword_set

# remove blanks
def blank_rm(blist:list):
    """去除list中的空格项/remove blank items in the list"""
    newlist=[]
    for item in blist:
       if (item != '') and (item!=' '):
           newlist.append(item)
    
    return newlist

# wordcut and clean text
def remove_punctuation(line):
    line = str(line)
    if line.strip()=='':
        return ''
    rule = re.compile(u"[^a-zA-Z0-9\u4E00-\u9FA5]")
    line = rule.sub('',line)
    return line

def cut_text(data):
    data['cut_text'] = data['clean_text'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) ]))
    return data

def sequence_vectorize(train_texts, val_texts):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > max_len:
        max_length = max_len

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer

# def get_length(data):
#     data['text_length'] = data['clean_text'].str.len()
#     return data
def data_split(data,split_fraction):
    train = data.sample(frac = split_fraction,random_state=0)
    test= data.drop(train.index)
    x_train = train.cut_text.tolist()
    x_test = test.cut_text.tolist()
    y_train = train.label
    y_test = test.label  
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape) 
    return x_train,y_train,x_test,y_test

def sequence_vectorize(train_texts, val_texts,max_words,max_len):
    """Vectorizes texts as sequence vectors.

    1 text = 1 sequence vector with fixed length.

    # Arguments
        train_texts: list, training text strings.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val, word_index: vectorized training and validation
            texts and word index dictionary.
    """
    # Create vocabulary with training texts.
    tokenizer = text.Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(train_texts)

    # Vectorize training and validation texts.
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # Get max sequence length.
    max_length = len(max(x_train, key=len))
    if max_length > max_len:
        max_length = max_len

    # Fix sequence length to max value. Sequences shorter than the length are
    # padded in the beginning and sequences longer are truncated
    # at the beginning.
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer


def cnn_model(vocab_size,embedding_dim):
  cnnmodel = keras.Sequential()

  cnnmodel.add(keras.layers.Embedding(vocab_size, embedding_dim))

  cnnmodel.add(keras.layers.Conv1D(filters=128,kernel_size=2,strides=1,kernel_initializer='he_normal',padding='VALID',activation='relu',name='conv'))

  cnnmodel.add(keras.layers.GlobalMaxPooling1D())
  cnnmodel.add(keras.layers.Dense(64, activation='relu'))

  cnnmodel.add(keras.layers.Dropout(0.3))
  cnnmodel.add(keras.layers.Dense(1, activation='sigmoid'))

  cnnmodel.summary()  
  return cnnmodel   

def train(mod,learning_rate,epochs,batch_size,train_data,train_label):
  pre,recall = tf.keras.metrics.Precision(thresholds=0.5),tf.keras.metrics.Recall(thresholds=0.5)
  # tp,fp = tf.keras.metrics.TruePositives(),tf.keras.metrics.TrueNegatives()

  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
  # Restrict TensorFlow to only use the first GPU

    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    mod.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=[pre,'accuracy'])
    history = mod.fit(train_data,
                    train_label,
                    epochs=epochs,
                    batch_size=batch_size * 8,
                    validation_split= 0.2)
  else:
    print('using cpu')
    mod.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=[pre,'accuracy'])
    history1 = mod.fit(train_data,
                    train_label,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split= 0.2)
  return mod

def convert_text_to_sequence(input_text):
    token = joblib.load('tokenzier_20210412_v3.pkl') #模型载入
    text_sequence = token.texts_to_sequences(input_text)
    pad_text_sequence = sequence.pad_sequences(text_sequence)
    return pad_text_sequence

def offline_test(data):
    data, dic_y_mapping = add_encode_variable(data, "label")
    cnn  = keras.models.load_model('cnn_model_20210412_v3.h5')
    data['clean_text'] = data['text'].apply(remove_punctuation)
    dataa = cut_text(data)
    sequence_data = convert_text_to_sequence(data['cut_text'])
    prediction = cnn.predict(sequence_data)
    yhat = predict_binary_class(prediction)
    y = data['label'].to_numpy()
    class_report(y,yhat,dic_y_mapping)

###################### run #################################################
stopword = load_stopword(dict_path)
df['clean_text'] = df['text'].apply(remove_punctuation)
df = cut_text(df)
x_train,y_train,x_test,y_test = data_split(df,split_fraction=split_fraction)
x_train,x_test,tokenizer = sequence_vectorize(x_train, x_test,max_words,max_length)
model_structure = cnn_model(vocab_size,embedding_dim)
model = train(model_structure,learning_rate,epochs,batch_size,x_train,y_train)


###################### evaluate #################################################
# prediction = cnnmodel.predict(x_test)
# yhat_cnn = predict_binary_class(prediction)
# class_report(y_test,yhat_cnn,dic_y_mapping)



###################### save ################################################
# model.save("cnn_model_20210412_v3.h5")
# joblib.dump(tokenizer,'tokenzier_20210412_v3.pkl') 

