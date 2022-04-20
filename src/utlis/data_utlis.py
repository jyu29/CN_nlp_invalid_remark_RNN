import os
import re
import time
import logging
import pandas as pd
from jinjasql import JinjaSql
from sqlalchemy import create_engine
import glob
import yaml
import json
import psycopg2
from utlis.general_utlis import *


def read_yml(file_path):
    """
    Read a local yaml file and return a python dictionary

    :param file_path: (string) full path to the yaml file
    :return: (dict) data loaded
    """

    with open(file_path) as f:
        yaml_dict = yaml.safe_load(f)

    return yaml_dict

def conn_to_postgresql(dbname,username,password,host,port):
    try:
            # Connect to an existing database
        connection = psycopg2.connect(database=dbname, user=username, password=password, \
                            host=host, port=port)
        cursor = connection.cursor()
        print("PostgreSQL server information")
        print(connection.get_dsn_parameters(), "\n")
    # Executing a SQL query
        cursor.execute("SELECT version();")
    # Fetch result–
        record = cursor.fetchone()
        print("You are connected to - ", record, "\n")

    except (Exception, Error) as error:
        print("Error while connecting to PostgreSQL", error)
    return cursor

def connect_to_postgresql_ali(username, password):
    try:
        # engine = create_engine('postgresql://' + user_name + ':' + password + '@10.50.8.227:60906/dw') \
        engine = create_engine('postgresql://' + username + ':'
                               + password + \
    '@dmt-pgs-ppd.pg.rds.aliyuncs.com:1921/dw')\
        .execution_options(autocommit=True)
        conn = engine.connect()
        print('Connection to postgresql success.')
    except ValueError:
        print('Connection error')
    return conn

def connect_to_postgresql_aws(user_name, password):
    try:
        engine = create_engine('postgresql://' + user_name + ':' + password + '@10.50.8.227:60906/dw') \
        .execution_options(autocommit=True)
        conn = engine.connect()
        print('Connection to postgresql success.')
    except ValueError:
        print('Connection error')
    return conn



# connect to postgresql using .sql file
def print_sql_query_from_file(conn, sql, params = {}):
    j = JinjaSql()
    f = open(sql, 'r')
    template = f.read().replace('}}', ' | sqlsafe }}')
    f.close()
    query = j.prepare_query(template, params)[0]
    print('the SQL script from file is:',query)


def execute_sql_query_to_pandas(connection,sql,params = {}):
# params: input parameters in sql query.default is none
# params = {'model_code': (8493534,8365103),'zone':u'Z024'}
    log(logger, "Running SQL queries from '{}' file".format(sql))
    tic = time.time()
    j = JinjaSql()
    f = open(sql, 'r')
    template = f.read().replace('}}', ' | sqlsafe }}')
    f.close()
    query = j.prepare_query(template, params)[0]
    dataframe = pd.read_sql_query(query,connection)
    print(dataframe.head())
    toc = time.time()
    runtime = toc-tic
    print('runtime is: {:.0f}mins {:.0f}seconds'.format( runtime// 60, runtime % 60))
    return dataframe




# def load_raw_remark_to_pandas(connection,sql,params = {}):
# # params: input parameters in sql query.default is none
# # params = {'model_code': (8493534,8365103),'zone':u'Z024'}
#     log(logger, "Running SQL queries from '{}' file".format(sql))
#     j = JinjaSql()
#     f = open(sql, 'r')
#     template = f.read().replace('}}', ' | sqlsafe }}')
#     f.close()
#     query = j.prepare_query(template, params)[0]
#     dataframe = pd.read_sql_query(query,connection)
#     toc = time.time()
#     runtime = toc-tic
#     print('runtime is: {:.0f}mins {:.0f}seconds'.format( runtime// 60, runtime % 60))
    
#     return dataframe


# def read_sql_save_parquet(folder_path,chunksize,sql,con):
# # folder_path = 'src/data/input/order'
#     count = 0
#     for chunk in pd.read_sql_query(sql , con, chunksize):
#         file_path = folder_path + '/part.%s.parquet' % (count)
#         chunk.to_parquet(file_path, engine='pyarrow')
#         count += 1


