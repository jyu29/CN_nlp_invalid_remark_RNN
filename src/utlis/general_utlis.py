import time
import logging
import matplotlib.pyplot as plt
from  sklearn.metrics import classification_report 
from sklearn.metrics import accuracy_score, confusion_matrix


# logger = logging.getLogger(__name__)


def get_cur_time():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def log(logger, msg, is_error=False):
    """
    Use logger if logging has been configured otherwise fallback to print()
    :param logger: (logging.logger) logger to use
    :param msg: (string) message to log
    :param is_error: (boolean) Optional. Default is False. If True message is logged in ERROR level
    """
    if logger.hasHandlers():
        if is_error:
            logger.error(msg)
        else:
            logger.info(msg)
    else:
        print(msg)
        
def plot_distributions(data, x, max_cat=20, top=None, y=None, bins=None, figsize=(10,5)):
    ## univariate
    if y is None:
        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(x, fontsize=15)
        ### categorical
        if data[x].nunique() <= max_cat:
            if top is None:
                data[x].reset_index().groupby(x).count().sort_values(by="index").plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            else:   
                data[x].reset_index().groupby(x).count().sort_values(by="index").tail(top).plot(kind="barh", legend=False, ax=ax).grid(axis='x')
            ax.set(ylabel=None)
        ### numerical
        else:
            sns.distplot(data[x], hist=True, kde=True, kde_kws={"shade":True}, ax=ax)
            ax.grid(True)
            ax.set(xlabel=None, yticklabels=[], yticks=[])

def plot_gradients(model_history):

    history_dict = model_history.history
    print(history_dict.keys())

    acc = history_dict['accuracy']
    val_acc = history_dict['val_accuracy']
    loss = history_dict['loss']
    val_loss = history_dict['val_loss']

    epochs = range(1, len(acc) + 1)
    fig = plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
# "bo" is for "blue dot"
    plt.plot(epochs, loss, 'r', label='Training loss')
# b is for "solid blue line"
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
# plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(epochs, acc, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()
    
def class_report(y,ypred,keys):
    labels = list(dic_y_mapping.values())

    labels_num = list(dic_y_mapping.keys())

    conf_mat = confusion_matrix(y, ypred)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.heatmap(conf_mat, annot=True, fmt='d',xticklabels=labels, yticklabels=labels_num,cmap="Blues")
    plt.ylabel('actual results',fontsize=12);
    plt.xlabel('predict result',fontsize=12)
    print('accuracy %s' % accuracy_score(y, ypred))
    print(classification_report(y, ypred,target_names=[str(w) for w in labels]))
    
def add_encode_variable(dtf, column):
    dtf[column+"_id"] = dtf[column].factorize(sort=True)[0]
    dic_class_mapping = dict( dtf[[column+"_id",column]].drop_duplicates().sort_values(column+"_id").values )
    return dtf, dic_class_mapping
# df, dic_y_mapping = add_encode_variable(df, "label_confirm")

def predict_binary_class(yhat):
    yhat[yhat<=0.5]=0
    yhat[yhat>0.5]=1
    return yhat


