'''processing and prediction '''
# author: jyu29
# create time: 2021-08

# read latest row number since last prediction
import pandas as pd
import psycopg2
from sqlalchemy import create_engine 
import jieba
import joblib
import re
import os
import warnings
import datetime
import keras
from tensorflow.python.keras.preprocessing import sequence,text
import time 
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
def warn(*args, **kwargs):
    pass
warnings.warn = warn
logging.getLogger('tensorflow').disabled = True



def read_max_seq():
    with open(r'/data/project/invalid_remark/max_seq_id.txt','r') as f:
        max_id=f.read()
#     print(max_id)
    return max_id

# query data from datamart
def load_channel_data_with_seq(start_id):
    conn = psycopg2.connect(database="dw", user="text_miner", password="gQ@8fwta2pFtZayxxi", host="10.50.8.227", port="60906")
    c= conn.cursor()
    sql="""select seq_id, customer_remark, "remark_Info" from smartdata_pro.f_invalide_customer_remark where seq_id>=%d;"""%start_id
    c.execute(sql)
    data=c.fetchall()
    #print(data)
    raw_remarks=[]
    for item in data:
        temp_list=list(item[0:2])
        temp_json=eval(item[2])[0]
        temp_list.extend([temp_json['commentsType'],temp_json['channel']])
        temp_tuple=tuple(temp_list)
        raw_remarks.append(temp_tuple)
    
    raw_remarks=pd.DataFrame(raw_remarks,columns=['seq_id','custom_mark','comments_Type','channel'])
    print(raw_remarks.head())
    #raw_remark=pd.DataFrame(c.fetchall(),columns=['seq_id','custom_mark','remark_info'])
    conn.close()
    
    return raw_remarks

# load stopwords
def load_stopword():
#     jieba.load_userdict(r"/data/jyu29/nlp/text classfication/ilsten/data/stopwordmaster.txt")
    jieba.load_userdict(r"/data/jyu29/nlp/text classfication/ilsten/data/dictionary/joe_remark_dict.txt")

    with open(r'/data/jyu29/nlp/text classfication/ilsten/data/stopword.txt',encoding='utf8') as f:
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



# load tokenizer and transform data into sequence data
def convert_text_to_sequence(input_text):
    token = joblib.load('/data/jyu29/nlp/text classfication/dataFile_0922.pkl')
    text_sequence = token.texts_to_sequences(input_text)
    pad_text_sequence = sequence.pad_sequences(text_sequence)
    
#     sequence_text = tokenized_text.texts_to_sequences(tokenized_text)
    return pad_text_sequence

# update prediction to datalake
def update_table(data): 
    input_data=[]
    for ind, items in data.iterrows():
        input_data.append(tuple(items))       
        
    conn = psycopg2.connect(database="dw", user="text_miner", password="gQ@8fwta2pFtZayxxi", host="10.50.8.227", port="60906")

    c= conn.cursor()
    
    c.execute("""
    update smartdata_pro.f_invalide_customer_remark as f
    set 
    invalid_remark=s.invalid_remark
    from unnest(%s) s(seq_id integer,invalid_remark integer)
    where f.seq_id=s.seq_id
    """,(input_data,))
    
    conn.commit()
    
    c.execute("""
              update smartdata_pro.f_invalide_customer_remark
set invalid_remark=0
where customer_remark like '%黑边%'""")
    conn.commit()
    conn.close()
    print('database is updated!')
    
def save_max_seq(max_seq):
    with open(r'/data/project/invalid_remark/max_seq_id.txt','w') as f:
        f.write(max_seq)
    print('Save max seq_id in file')
    print('-------------------------------------------------------------------------')
#     print(max_seq)


#     with open(r'/data/project/invalid_remark/max_seq_id.txt','r') as f:




def predict_binary_class(yhat):
    yhat[yhat<=0.5]=0
    yhat[yhat>0.5]=1
    return yhat


def user_comment_predict(data):
    rnnmodel = keras.models.load_model('/data/jyu29/nlp/text classfication/invalid_remark/model/rnnmodel_0922.h5')
#     print ('Model loaded')
            #query = query.reindex(columns=model_columns, fill_value=0)
    stopword = load_stopword()
    data = data.fillna('谢谢老板')
    data['clean_text'] = data['custom_mark'].apply(remove_punctuation)

    data['cut_text'] = data['clean_text'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopword]))
    print('clean_text:',data['cut_text'].values)
    sequence_data = convert_text_to_sequence(data['cut_text'])
    
    try:
    
        prediction = rnnmodel.predict(sequence_data)
        yhat = int(predict_binary_class(prediction))
        data['invalid_remark'] = yhat
    except Exception as e:
        data['invalid_remark'] = 1
    
    return data[['seq_id','invalid_remark']]

# def customer_service_predict(data):
#     rnnmodel = keras.models.load_model('/data/jyu29/nlp/text classfication/invalid_remark/model/rnnmodel_0922.h5')
# #     print ('Model loaded')
#             #query = query.reindex(columns=model_columns, fill_value=0)
#     stopword = load_stopword()
#     data = data.fillna('谢谢老板')
#     data['clean_text'] = data['custom_mark'].apply(remove_punctuation)

#     data['cut_text'] = data['clean_text'].apply(lambda x: " ".join([w for w in list(jieba.cut(x)) if w not in stopword]))
#     print('clean_text:',data['cut_text'].values)

#     sequence_data = convert_text_to_sequence(data['cut_text'])
#     try:
    
#         prediction = rnnmodel.predict(sequence_data)
#         yhat = int(predict_binary_class(prediction))
#         data['invalid_remark'] = yhat
#     except Exception as e:
#         data['invalid_remark'] = 1
#     return data[['seq_id','invalid_remark']]


def predict_all_kinds(data):
    """devide data into two groups user comment and customer service comment;
    try to predict by seperate models then merge them """
#     data_customer=data[data['comments_Type']==1]
#     data_service=data[data['comments_Type']==2]
    if len(data)==0:
        print('no user comment update,the run time is ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        result_user=pd.DataFrame(columns=['seq_id','invalid_remark'])
    else:
        result_user=user_comment_predict(data)
    
#     if len(data_service)==0:
#         print('No customer service update, the runtime is ',datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
#         result_customer=pd.DataFrame(columns=['seq_id','invalid_remark'])
#     else:
#         result_customer=customer_service_predict(data_service)
        #print(result_customer)
    
#     result_all=result_user.append(result_customer,ignore_index=True)
    result_all = result_user
#     result_all.reset_index(drop=True,inplace=True)
    result_all.seq_id = result_all.seq_id.astype(int)
    result_all.invalid_remark = result_all.invalid_remark.astype(int)

    
    return result_all
    



def predict_and_update():
    max_seq_id=read_max_seq()
#     print(max_seq_id)
    start_id=int(max_seq_id)+1
 
    try:
        query=load_channel_data_with_seq(start_id=start_id)
#         print(query)
        data_len=len(query)
    
        if data_len>0:
            final_result=predict_all_kinds(query)
            print(final_result)
            max_id=final_result['seq_id'].max()
                   
            update_table(final_result)
            save_max_seq(str(max_id))
            
    except Exception as e:
        print(str(e))
                
            
        
def timer(n):
    while True:
        predict_and_update()
#         print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        time.sleep(n)
    
if __name__=='__main__':
    
    warnings.filterwarnings('ignore')
    print('the process id:',os.getpid())
    timer(10)
    
    
        
        
    