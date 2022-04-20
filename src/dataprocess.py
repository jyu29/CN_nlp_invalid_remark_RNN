import pandas as pd
from utlis.data_utlis import *


def preprocessing(data):
    '''data: panda dataframe'''
    remark_info_str_split = data.remark_info.str.split(':',4,True)
    channel = remark_info_str_split[1].str.split(',',2,True)
    comments_Type = remark_info_str_split[2].str.split(',',2,True)
    data['channel'] = channel[0]
    data['comments_Type'] = comments_Type[0]
    new_data = data[['seq_id','customer_remark','channel','comments_Type']]
    return new_data


def correct_label_for_train_data(data,invaild_keyword,valid_keyword):
# invalid keyword（1）
# -|AG  
# -价格
# -全渠道类型（自提，门店自提，同城购，定时送）
# -尽快发货
# valid keyword（0）
# 有效
# 锁单   
# 扣单
    data['label_correct'] = data['label_confirm']

# seach for text contains keyword
    data["invalid_rulecheck"] = data['text'].apply(lambda x: 'match' if any(i 
                                   in x for i in invalid_keyword) else 'nomatch')
    data["valid_rulecheck"] = data['text'].apply(lambda x: 'match' if any(i 
                                   in x for i in valid_keyword) else 'nomatch')
# dfold.loc[(dfold['label_confirm'] == 0) & (dfold['invalid_rulecheck'] == 'match')]

# replace label from valid to invalid
    data.loc[(data['label_confirm'] == 0) & \
          (data['invalid_rulecheck'] == 'match'),'label_correct']=1

# replace label from invalid to valid
    data.loc[(dfold['label_confirm'] == 1) & 
          (data['valid_rulecheck'] == 'match'),'label_correct']=0
    data = data[['text','label_correct']]
    data = data.rename({'label_correct':'label'},axis = 1).drop_duplicates(subset ="text")
    return data

##################

def add_new_data_to_train(data_from_db,invalid_keyword,valid_keyword):
# AG：1 - 347
# 扣单：3641
# 锁单：0 - 100
# 缺货：62 - 17353
# 余单撤销：10 - 902 
# 各一件：12 - 212
# 自提：32 - 582
# 同城购： 4 -63
    df = data_from_db.loc[~data_from_db.customer_remark.isnull()]
    df = df[['customer_remark']]
    df = df.rename({'customer_remark':'text'},axis=1)
    df["invalid_add"] = df['text'].apply(lambda x: 'match' if any(i 
                    in x for i in invalid_keyword) else 'nomatch') 
    df["valid_add"] = df['text'].apply(lambda x: 'match' if any(i 
                    in x for i in valid_keyword_v3) else 'nomatch') 

    invalid_newdata = df.loc[df.invalid_add=='match'].drop_duplicates(subset ="text")
    invalid_newdata['label']=1
    valid_newdata = df.loc[df.valid_add=='match'].drop_duplicates(subset ="text")
    valid_newdata['label']= 0 
    newdata = valid_newdata.append(invalid_newdata)
    newdata = newdata[['text','label']]
    return newdata

# config
config = read_yml('/Users/JOE/Desktop/dev/invalid_remark/config/dev.yml')
raw_data_path = config['remote_data_path']['raw']
preproces_data_path = config['remote_data_path']['preproces']
process_data_path = config['remote_data_path']['processed']
invalid_keyword = config['business_rule']['invalid_keyword']
valid_keyword= config['business_rule']['valid_keyword']
valid_keyword_v3= config['business_rule']['valid_keyword_v3']

# read data - change path later
dfold = pd.read_csv(preproces_data_path +'invalid_remark_raw_masterdata_v2.csv')
df = pd.read_parquet(preproces_data_path +'raw_remarks.gzip')

# processing
dfold = correct_label_for_train_data(dfold,invalid_keyword,valid_keyword)
df = add_new_data_to_train(df,invalid_keyword,valid_keyword_v3)
df = df.loc[df['text'].str.contains('缺货'),'label']=0
master  = pd.concat([df,dfold]).drop_duplicates(subset ="text")

# final check
# master.loc[master["text"].str.contains('AG')].label.unique()
master.loc[master["text"].str.contains('AG'),'label']=1
master.loc[master["text"].str.contains('自提'),'label']=1
master.loc[master["text"].str.contains('邮政'),'label']=0
master.loc[master["text"].str.contains('EMS'),'label']=0
master.loc[master["text"].str.contains('圆通'),'label']=0
master.loc[master["text"].str.contains('顺丰'),'label']=0
master.loc[master["text"].str.contains('申通'),'label']=0
master.loc[master["text"].str.contains('百世'),'label']=0
master.loc[master["text"].str.contains('中通'),'label']=0
master.loc[master["text"].str.contains('汇通'),'label']=0


# save 
master.to_parquet(process_data_path+'invalid_remark_raw_masterdata_v3.gzip',engine='auto',compression='gzip')







  