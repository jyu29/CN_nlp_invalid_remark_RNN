import pandas as pd
from utlis.data_utlis import *
import logging

# config
config = read_yml('config/dev.yml')
username = config['dmart_textminer']['username']
password = config ['dmart_textminer']['password']
max_seq = config ['sql_parameter']['seq_id']
save_folder_path = config['remote_data_path']['raw']
#seq_id = modify later

# cnnect to alidb
conn = connect_to_postgresql_ali(username, password)

# #sql file
sql_remark = 'src/data/sql/rmk_seq.sql'

# #execute query to pandas
# print_sql_query_from_file(conn,sql_remark,params = {'seq_id': seq_id})
df = execute_sql_query_to_pandas(conn,sql=sql_remark,params = {'seq_id': seq_id}


# save df to diretory
df.to_parquet(save_folder_path+'raw_remarks.gzip',engine='auto',compression='gzip')