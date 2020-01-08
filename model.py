import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

###visulation with graphs using seaborn###
class Visulation:
    def HeatMap(self,data):
        sns.heatmap(data.isnull(),yticklabels=False,cbar=False,cmap='viridis')
        plt.show()
    

###Anylasis the dataset###
def Anylasis(data):
    print('\n',data.head())
    print('\n\n\n',data.shape)
    print('\n\n\n',data.columns)
    print('\n\n\n',data.info())
    print('\n\n\n',data.corr())
    print('\n\n\n',data.isnull().sum(axis=0))
    print('\n\n\n',data.nunique())
    print('\n\n\n',data['car'].unique())


###Encoding Categorical Data###
def DataEncoding(data):
    from sklearn.preprocessing import LabelEncoder
    lb1 = LabelEncoder()
    data['buying'] = lb1.fit_transform(data['buying'])

    lb2 = LabelEncoder()
    data['maint'] = lb2.fit_transform(data['maint'])

    lb3 = LabelEncoder()
    data['doors'] = lb3.fit_transform(data['doors'])

    lb4 = LabelEncoder()
    data['persons'] = lb4.fit_transform(data['persons'])

    lb5 = LabelEncoder()
    data['lug_boot'] = lb5.fit_transform(data['lug_boot'])

    lb6 = LabelEncoder()
    data['safety'] = lb6.fit_transform(data['safety'])

    lb7 = LabelEncoder()
    data['car'] = lb7.fit_transform(data['car'])
    return data

###Using Decision Tree Algorithm###
def DecisionTree(data):
    from sklearn.model_selection import train_test_split
    X = data.drop('car',axis=1)
    y = data['car']
    ##Splitting dataset into train and test set##
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

    ##Decision Tree with the help of Scikit-learn##
    from sklearn.tree import DecisionTreeClassifier
    dtree = DecisionTreeClassifier()
    dtree.fit(X_train, y_train)

    ##Predicting test set data##
    predictions = dtree.predict(X_test)
    
    ##For checking accuracy and efficiency##
    from sklearn.metrics import classification_report, confusion_matrix
    print('\n#####Decision Tree#####')
    print('\n\nConfusion Matrix:\n',confusion_matrix(y_test,predictions))
    print('\n\n\nClassification Report:\n',classification_report(y_test,predictions))


###Using Random Forest Algorith###
def RandomForest(data):
    from sklearn.model_selection import train_test_split
    X = data.drop('car',axis=1)
    y = data['car']
    ##Splitting dataset into train and test set##
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3)

    ##Random Forest with the help of Scikit-learn##
    from sklearn.ensemble import RandomForestClassifier
    rfc = RandomForestClassifier(n_estimators=200)
    rfc.fit(X_train, y_train)

    ##Predicting test set data##
    predictions = rfc.predict(X_test)

    ##For checking accuracy and efficiency##
    from sklearn.metrics import classification_report, confusion_matrix
    print('\n#####Random Forest#####')
    print('\n\nConfusion Matrix:\n',confusion_matrix(y_test,predictions))
    print('\n\n\nClassification Report:\n',classification_report(y_test,predictions))


def main():
    ###Reading Dataset###
    dataset = pd.read_csv("cars_dataset.csv")

    ###Heatmap to check null values in dataset###
    v1 = Visulation()
    v1.HeatMap(dataset)

    Anylasis(dataset)
    dataset = DataEncoding(dataset)
    DecisionTree(dataset)
    RandomForest(dataset)

if __name__ == "__main__":
    main()
