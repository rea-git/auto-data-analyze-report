import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
data = pd.read_csv('data.csv')
"""data_descrip = data.describe()
categories = ['count','mean','std','min','25%','50%','75%','max']"""
"""
def cleaning(data):
    data = data.dropna() # remove empty 
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = pd.to_numeric(data[col],errors="ignore")
    return data
new_data = cleaning(data)
new_data_descrip = new_data.describe()

"""
{'index': {'count': 5.0, 'mean': 6.4, 'std': 3.5777087639996634, 'min': 1.0, '25%': 5.0, '50%': 7.0, '75%': 9.0, 'max': 10.0}, 
'col1': {'count': 5.0, 'mean': 3.4, 'std': 2.8809720581775866, 'min': 1.0, '25%': 1.0, '50%': 3.0, '75%': 4.0, 'max': 8.0}, 
'col2': {'count': 5.0, 'mean': 4.0, 'std': 1.5811388300841898, 'min': 2.0, '25%': 3.0, '50%': 4.0, '75%': 5.0, 'max': 6.0}, 
'col3': {'count': 5.0, 'mean': 4.2, 'std': 1.9235384061671346, 'min': 2.0, '25%': 3.0, '50%': 4.0, '75%': 5.0, 'max': 7.0}}
"""
new_data_descrip = new_data_descrip.to_dict()
data_descrip = data_descrip.to_dict()

def bar_graph(old_data_raw,new_data_raw,title,width,categories):
    old_data=[]
    new_data=[]
    for i in old_data_raw:
        old_data.append(int(old_data_raw[i][title]))
        new_data.append(int(new_data_raw[i][title]))
    print(title,old_data)
    print(title,new_data)
    plt.figure()
    x=np.arange(len(categories))
    plt.bar(x-width/2,old_data,width,label ='before' )
    plt.bar(x+width/2,new_data,width,label ='after' ) 
    plt.xticks(x,categories)
    #plt.xlabel(xlabel)
    #plt.ylabel(ylabel)   
    plt.title(title)
    plt.legend()
    plt.savefig(f'images/{title}.png')
    plt.clf()
#count
bar_graph(data_descrip,new_data_descrip,'count',0.4,data_descrip.keys())
#mean
bar_graph(data_descrip,new_data_descrip,'mean',0.4,data_descrip.keys())
#std
bar_graph(data_descrip,new_data_descrip,'std',0.4,data_descrip.keys())
#min
bar_graph(data_descrip,new_data_descrip,'min',0.4,data_descrip.keys())
#25%
bar_graph(data_descrip,new_data_descrip,'25%',0.4,data_descrip.keys())
#50%
bar_graph(data_descrip,new_data_descrip,'50%',0.4,data_descrip.keys())
#75%
bar_graph(data_descrip,new_data_descrip,'75%',0.4,data_descrip.keys())
#max
bar_graph(data_descrip,new_data_descrip,'max',0.4,data_descrip.keys())"""
from sklearn.neighbors import KNeighborsClassifier

def knn_predict(data):
    null_data  =data[data.isnull().any(axis=1)]
    nonull_data = data.dropna()

    points = list(nonull_data.itertuples(index=False, name=None))
    newdata = list(null_data.itertuples(index=False, name=None))
    for row in null_data:
        



    

