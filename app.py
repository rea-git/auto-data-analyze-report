from flask import Flask, request, redirect, url_for,render_template
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
app = Flask(__name__)

@app.route('/',methods = ['POST','GET'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        data = pd.read_csv(request.files['data'])
        missing_method = request.form.get('missing_method')
        outlier_method = request.form.get('outlier_method')
        before_descrip = data.describe()
        #cleaning
        def encoding(data):
            encoder_mapping = {}
            for col in data.select_dtypes(include=['object']).columns:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
                encoder_mapping[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
            data = data.apply(pd.to_numeric,errors='coerce')
            return data, encoder_mapping
        def missing_value(data,method):
            if method == 'mean':
                data = data.fillna(data.mean())
            if method == 'median':
                data = data.fillna(data.median())
            if method == 'mode':
                data = data.fillna(data.mode().iloc[0])
            if method == 'KNN':
                imputer = KNNImputer(n_neighbors=2)
                data = pd.DataFrame(imputer.fit_transform(data),columns=data.columns)
            if method == 'remove':
                data = data.dropna() # remove empty 
            return data
        def outlier(data,method):
            if method =='winsorize':
                for col in data.select_dtypes(include=['number']).columns:
                    data[col] =winsorize(data[col], limits=[0.05, 0.05])
            return data
        new_data,encoder_mapping = encoding(data)
        new_data = missing_value(new_data,missing_method)
        new_data = outlier(new_data,outlier_method)
        after_descrip = new_data.describe()
        diff_descrip = before_descrip-after_descrip

        bef_html = before_descrip.to_html(classes="table table-bordered", border=1)
        aft_html = after_descrip.to_html(classes="table table-bordered", border=1)
        diff_html = diff_descrip.to_html(classes="table table-bordered", border=1)
        """
        {'index': {'count': 5.0, 'mean': 6.4, 'std': 3.5777087639996634, 'min': 1.0, '25%': 5.0, '50%': 7.0, '75%': 9.0, 'max': 10.0}, 
        'col1': {'count': 5.0, 'mean': 3.4, 'std': 2.8809720581775866, 'min': 1.0, '25%': 1.0, '50%': 3.0, '75%': 4.0, 'max': 8.0}, 
        'col2': {'count': 5.0, 'mean': 4.0, 'std': 1.5811388300841898, 'min': 2.0, '25%': 3.0, '50%': 4.0, '75%': 5.0, 'max': 6.0}, 
        'col3': {'count': 5.0, 'mean': 4.2, 'std': 1.9235384061671346, 'min': 2.0, '25%': 3.0, '50%': 4.0, '75%': 5.0, 'max': 7.0}}
        """
        old_data_raw = before_descrip.to_dict()
        new_data_raw = after_descrip.to_dict()
        def make_graph(old_data_raw,new_data_raw):
            
            old_data={}
            new_data={}
            for i in old_data_raw:
                old_data[i]=old_data_raw[i].values()
                new_data[i]=new_data_raw[i].values()
            categories = old_data_raw[i].keys()
            print(old_data)
            print(new_data)
            for i in old_data:
                plt.figure(figsize=(3.5,2.5))
                x=np.arange(len(categories))
                plt.plot(old_data[i],label ='before',color='red' )
                plt.plot(new_data[i],label ='after',color='blue' ) 
                plt.xticks(x,categories)
                #plt.xlabel(xlabel)
                #plt.ylabel(ylabel)
                title = i  
                plt.title(title)
                plt.legend()
                plt.savefig(f'static/images/{title}.png', dpi=120)
                plt.clf()
        make_graph(old_data_raw,new_data_raw)
        return render_template('output.html',bef_html = bef_html, aft_html = aft_html,diff_html=diff_html,disc=old_data_raw.keys())
app.run(debug=True)