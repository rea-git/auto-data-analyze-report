from flask import Flask, request, redirect, url_for,render_template
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
from scipy.stats.mstats import winsorize
import seaborn as sns
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
        def to_numeric(data):
            for col in data.columns:
                numeric_count = pd.to_numeric(data[col], errors='coerce').notna().sum()
                total_count = len(data[col])

                # If majority are numbers, convert whole column to numeric
                if numeric_count / total_count > 0.8:  # threshold can be tuned
                    # Replace non-numeric (like 'Nil') with 0 before converting
                    data[col] = pd.to_numeric(data[col], errors='coerce').fillna(np.nan)
                    # Step 2: Label encode only remaining object columns
            return data
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
                numeric_cols = data.select_dtypes(include=['number']).columns
                imputer = KNNImputer(n_neighbors=2)
                imputed_array = imputer.fit_transform(data[numeric_cols])
                imputed_df = pd.DataFrame(imputed_array, columns=numeric_cols, index=data.index)
                data[numeric_cols] = imputed_df  # assign back safely
            if method == 'remove':
                data = data.dropna() # remove empty 
            return data
        def outlier(data,method):
            if method =='winsorize':
                for col in data.select_dtypes(include=['number']).columns:
                    data[col] =winsorize(data[col], limits=[0.05, 0.05])
            return data
        new_data = data.copy()
        new_data = to_numeric(new_data)
        new_data,encoder_mapping = encoding(new_data)

        new_data = new_data.dropna(axis=1, how='all')
        new_data = new_data.loc[:, ~new_data.columns.str.contains('^Unnamed')]

        new_data = missing_value(new_data,missing_method)
        new_data = outlier(new_data,outlier_method)
        data_descrip = data.describe()
        new_data_descrip = new_data.describe()
        # Table headers
        html_output = "<table border='1'><tr><th>Column</th><th>Category</th><th>Encoded</th></tr>"
        
        # Loop through columns and their mappings
        for col, mapping in encoder_mapping.items():
            for category, code in mapping.items():
                html_output += f"<tr><td>{col}</td><td>{category}</td><td>{code}</td></tr>"
        #margin of error
        z_scores = {
            '90%':1.645,
            '95%':1.96,
            '99%':2.575
        }
        n= new_data_descrip.loc['count']
        std = new_data_descrip.loc['std']
        margin_errors = {}
        for level, z in z_scores.items():
            margin_errors[level] = z * (std / np.sqrt(n))
        margin_errors_df = pd.DataFrame(margin_errors)

        html_output += "</table>"
        numeric_data = new_data.select_dtypes(include=['number'])
        corr = numeric_data.corr()

        plt.figure(figsize=(20,16))
        plt.tight_layout() 
        sns.heatmap(corr, annot=True, cmap='coolwarm')
        plt.savefig('static/images/corr.png', bbox_inches='tight')
        plt.close()
    return render_template('output.html',margin_of_error = margin_errors_df.to_html(), encoding_mapping = html_output,missing_method = missing_method,outlier_method = outlier_method,bef_html = data_descrip.to_html(), aft_html = new_data_descrip.to_html())
app.run(debug=True)