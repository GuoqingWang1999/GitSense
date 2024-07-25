import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler

def select_feature():
    
    data = pd.read_csv('your_dic/your_data_name',index_col=0)
    
    print(len(data))
    # read all features
    X = data.iloc[:, 4:-1] 
    y = data.iloc[:, -1]   
    y = [1 if item == 0 else 0 for item in y]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    smote = SMOTE()
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
    # we simply use the default parameters
    model = xgb.XGBClassifier()
    model.fit(X_train_resampled, y_train_resampled)


    y_pred = model.predict(X_test_scaled)

    # calculate feature importance
    importance = model.feature_importances_

    for i, imp in enumerate(importance):
        print(f"Feature {i+1}: Importance = {imp}")

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
    

if __name__ == '__main__':
    select_feature()