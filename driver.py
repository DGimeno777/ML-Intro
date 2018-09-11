# Import Libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def main():
    dataset = load_data()
    print(dataset)

'''
Return data from the UCI Machine Learning repository
'''
def load_data():
    # Location to load data from
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # Array of names of data columns to query from the database
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    # Returns every data piece in names from the data set at the url
    return pandas.read_csv(url, names=names)

main()