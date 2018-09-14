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
    show_dataset_histogram(dataset)

'''
Creates plots of each quantifiable data column of the given dataset
'''
def show_dataset_plot(dataset):
    # Plots the dataset on the plt library
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    # Shows the plot of the plt library
    plt.show()

'''
Creates histogram of each quantifiable data column in the dataset
'''
def show_dataset_histogram(dataset):
    dataset.hist()
    plt.show()

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