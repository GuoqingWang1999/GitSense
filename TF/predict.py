from sklearn.feature_extraction.text import CountVectorizer
import json
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import glob
import math
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import precision_score, recall_score, f1_score
import time
def predict():
    
    alldata = json.load(open('data.json'))

    
    vectorizer = CountVectorizer(max_features=10)  
    vectorizer.fit(alldata)

    
    unknown_token = "<UNK>"

    # please consider program language
    # # 'auto', 'break', 'case', 'char', 'const', 'continue', 'default', 'do', 
    # # 'double', 'else', 'enum', 'extern', 'float', 'for', 'goto', 'if', 'int', 
    # # 'long', 'register', 'return', 'short', 'signed', 'sizeof', 'static', 
    # # 'struct', 'switch', 'typedef', 'union', 'unsigned', 'void', 'volatile', 'while'
    # # 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 
    # # 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 
    # # 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 
    # # 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield'

    keywords = ['abstract', 'assert', 'boolean', 'break', 'byte', 'case', 'catch', 'char', 
    'class', 'const', 'continue', 'default', 'do', 'double', 'else', 'enum', 
    'extends', 'final', 'finally', 'float', 'for', 'goto', 'if', 'implements', 
    'import', 'instanceof', 'int', 'interface', 'long', 'native', 'new', 'null', 
    'package', 'private', 'protected', 'public', 'return', 'short', 'static', 
    'strictfp', 'super', 'switch', 'synchronized', 'this', 'throw', 'throws', 
    'transient', 'try', 'void', 'volatile', 'while', 'true', 'false']

    tokenized_lines = []
    for data in alldata:
        words = data.split('\n')
        line = []
        tokenized_line = []
        for word in words:
            line = line + word.split()
            for token in line:
                if token in keywords:
                    tokenized_line.append(token)
                elif token in vectorizer.vocabulary_:
                    tokenized_line.append(token)     
        

        tokenized_lines.append(' '.join(tokenized_line))
    
    # tokenized_lines = []
    # for line in alldata:
    #     words = line.split()
    #     tokenized_line = []
    #     for word in words:
    #         if word in keywords:

    #             tokenized_line.append(word)
    #         elif word in vectorizer.vocabulary_:
               
    #             tokenized_line.append(word)
    #         else:
    #             tokenized_line.append(unknown_token)
    #     tokenized_lines.append(' '.join(tokenized_line))

    
    feature_vectors = vectorizer.transform(tokenized_lines).toarray()

    df = pd.read_csv('alldata.csv')
    labels = df['now_label'].map({0: 1, 1: 0})  

    
    X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.25, shuffle=False)

    
    clf = RandomForestClassifier(n_estimators=100)

    ros = RandomOverSampler(sampling_strategy=0.80,random_state=0)

    X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

    clf.fit(X_resampled, y_resampled)
    
    predicted_labels = clf.predict(X_test)
    
    tn, fp, fn, tp = confusion_matrix(y_test, predicted_labels).ravel()

    print(f'TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}')

    precision = precision_score(y_test, predicted_labels)
    recall = recall_score(y_test, predicted_labels)
    f1 = f1_score(y_test, predicted_labels)

    print(f'Precision: {precision}, Recall: {recall}, F1: {f1}')

    precision_0 = precision_score(y_test, predicted_labels, pos_label=0)
    recall_0 = recall_score(y_test, predicted_labels, pos_label=0)
    f1_0 = f1_score(y_test, predicted_labels,pos_label=0)
    print(f'Precision for label 0: {precision_0}, Recall for label 0: {recall_0}, F1 for label 0: {f1_0}')

if __name__ == '__main__':
    predict()