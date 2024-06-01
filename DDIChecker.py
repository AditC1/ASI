import os
import glob
import pickle
import copy
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelBinarizer

def make_prediction(output_dir, output_file, ddi_input_file, trained_model, threshold):  
    with open('./data/multilabelbinarizer.pkl', 'rb') as fid:
        lb = pickle.load(fid)
    
    df = pd.read_csv(ddi_input_file, index_col=0)    
    ddi_pairs = list(df.index)
    X = df.values    
    model = load_model(trained_model)
    y_predicted = model.predict(X)    
    original_predicted_ddi = copy.deepcopy(y_predicted)
    y_predicted[y_predicted >= threshold] = 1
    y_predicted[y_predicted < threshold] = 0

    y_predicted_inverse = lb.inverse_transform(y_predicted)   
    
    with open(output_file, 'w') as fp:
        fp.write('Drug pair\tPredicted class\tScore\n')
        for i in range(len(ddi_pairs)):
            predicted_ddi_score = original_predicted_ddi[i]
            predicted_ddi = y_predicted_inverse[i]
            each_ddi = ddi_pairs[i]           
            for each_predicted_ddi in predicted_ddi:
                fp.write('%s\t%s\t%s\n' % (each_ddi, each_predicted_ddi, predicted_ddi_score[each_predicted_ddi-1]))


