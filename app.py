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
        #cleaning
        def encoding(data):
            encoder_mapping = {}
            for col in data.select_dtypes(include=['object']).columns:
                encoder = LabelEncoder()
                data[col] = encoder.fit_transform(data[col])
                encoder_mapping[col] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
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
        new_data = data.copy()
        for col in new_data.columns:
            if pd.to_numeric(new_data[col], errors='coerce').notna().sum() == len(new_data[col]):
                new_data[col] = pd.to_numeric(new_data[col])
        new_data,encoder_mapping = encoding(new_data)
        new_data = missing_value(new_data,missing_method)
        new_data = outlier(new_data,outlier_method)
        data_descrip = new_data.describe()
        new_data_descrip = data.describe()
        change_data_descrip = change_data.describe()
        # Table headers
        html_output = "<table border='1'><tr><th>Column</th><th>Category</th><th>Encoded</th></tr>"
        
        # Loop through columns and their mappings
        for col, mapping in encoder_mapping.items():
            for category, code in mapping.items():
                html_output += f"<tr><td>{col}</td><td>{category}</td><td>{code}</td></tr>"
        
        html_output += "</table>"

    return render_template('output.html',encoding_mapping = html_output,missing_method = missing_method,outlier_method = outlier_method,bef_html = data_descrip.to_html(), aft_html = new_data_descrip.to_html())

app.run(debug=True)