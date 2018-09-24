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

    # Split dataset into the training data and the validation data
    # We will use the training data to train our ML model and the
    # validation data to see how the model fairs when trying to
    array = dataset.values

    # Data points of Flower size/dimensions (ex. sepal-length = flower leaf length)
    flower_parameters = array[:,0:4]

    # Array of answers to question: Given dimensions, what flower is it?
    flower_answers = array[:,4]

    # Want to use 80% of data we have to train model, 20% will be used to validate
    validation_size = 0.20

    # This will usually be randomly generated but we're going to hardcode it so we get the same results as the tutorial
    seed = 7

    # Splits the
    fp_train, fp_validation, fa_train, fa_validation = model_selection.train_test_split(flower_parameters,
                                                                                        flower_answers,
                                                                                        test_size=validation_size,
                                                                                        random_state=seed)

    # Create Test harness
    # We will be using a 10-fold cross validation
    # The CV will partition the test data into segments (10 in this case) to model the data and
    # (more on this soon)
    scoring = 'accuracy'

    # Spot Check Algorithms
    # Add model types to our model array
    models = []

    # Linear Algorithms
    # - Take a set number of inputs and create a linear relationship to model a correlation
    # - Model based off how likely things occurred together in the past, second order linear models include information
    #   about indirect relationships and can make inferences based off these co-occurences
    # - Can perform a binary classification which tells if the output is positive or negative
    # - Linear modeling cannot incorporate when the order of events matters (a -> b vs b -> a)
    models.append(('LR', LogisticRegression()))
    models.append(('LDA', LinearDiscriminantAnalysis()))

    # Non-Linear Algorithms
    # - Uses a non-linear model instead of a linear model which are more powerful but harder to train
    # - More computationally intensive that linear
    # - Allows for more complicated algorithms to be solved
    # - Most problems can be solved by using linear and non-linear should be used on ones that can't (to save power)
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB()))
    models.append(('SVM', SVC()))
    # evaluate models in turn
    results = []
    names = []
    # For each model, do the kfold, calculate the results, and append results to array
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, fp_train, fa_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)

    # Now choose model that got it correct the most


def show_training_results(names, results):
    """
    Shows the results of the trained algorithms in a boxplot
    :param names:
    :param results:
    :return:
    """
    # Make graph to compare algs
    fig = plt.figure()
    # Give graph a name
    fig.suptitle('Algorithm Comparison')
    # Add subplot with dimensions of 1x1 as first subplot
    ax = fig.add_subplot(111)
    # Create a boxplot from the training results
    plt.boxplot(results)
    # Sets x ticks to names of algorithms
    ax.set_xticklabels(names)
    # Show plot
    plt.show()

def show_dataset_plot(dataset):
    """
    Creates plots of each quantifiable data column of the given dataset
    :param dataset: dataset to show the plot of
    :return:
    """
    # Plots the dataset on the plt library
    dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
    # Shows the plot of the plt library
    plt.show()


def show_dataset_histogram(dataset):
    """
    Creates histogram of each quantifiable data column in the dataset
    """
    dataset.hist()
    plt.show()


def show_scatter_plot_matrix(dataset):
    """
    Creates a scatter plot metrix of the given dataset
    """
    scatter_matrix(dataset)
    plt.show()


def load_data():
    """
    Return data from the UCI Machine Learning repository
    """
    # Location to load data from
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    # Array of names of data columns to query from the database
    names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
    # Returns every data piece in names from the data set at the url
    return pandas.read_csv(url, names=names)

main()