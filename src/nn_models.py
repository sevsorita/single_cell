import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.layers as kl
import tensorflow.keras.losses as kls
import tensorflow.keras.optimizers as ko
import tensorflow.keras.models as km
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

import pickle
import os.path
from os import path

plt.rcParams.update({'font.size': 10, 'figure.figsize': (4.7747, 4.7747)})


class FFNN:
    """Adds utility funcionts for training, testing and analysis to a keras model

    :param model: a Keras model class
    :param optimizer: an instance of a optimizer class 
    """
    def __init__(self, model, optimizer, use_min_pred=False, model_number=None):
        self.model = model
        self.train_loss = []
        self.val_loss = []
        self.optimizer = optimizer
        self.using_min_pred = use_min_pred
        self.compiled = False

        if model_number is None:
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            self.model_number = current_time
        else:
            self.model_number = model_number

    def recompile(self):
        """Reinitialize weights and biases
        """
        self.model.compile(optimizer=self.optimizer, loss="mse")
        self.train_loss = []
        self.val_loss = []
        self.compiled = True

    def set_eta(self, eta):
        K.set_value(self.model.optimizer.learning_rate, eta)

    def save_model(self, path="/"):
        self.model.save(path)

    def load_model(self, path=None):
        if path==None:
            print("You must specify a model")
        else: 
            self.model = keras.model.load_model(path)

    def load_best_weights(self):
        self.model.load_weights(self.filename)

    def set_trainingset(self, x_train, y_train, datasetname="_"):
        """Set the training data

        :param x_train: training input
        :param y_train: training output
        """
        self.x_train = x_train
        self.y_train = y_train
        self.datasetname = datasetname

    def set_testingset(self, x_test, y_test):
        """Set the test data

        :param x_test: testing input
        :param y_train: testing output
        """
        self.x_test = x_test
        self.y_test = y_test

    def set_min_pred(self, min_val=None):
        if min_val is None:
            self.min_pred = self.y_train.min()
        else:
            self.min_pred = min_val
        self.using_min_pred = True


    def train(self, batch_size=5, epochs=10, verbose=0, val_split=0.2, early_stopping=False, checkpoint=False, baseline=1000, fold_nr=0):
        """ Fit the model on the training set

        :param batch_size: number of data points in each gradient descent step
        :param epochs: number of epochs to train for
        :param verbose: set to 2 or 3 to display training progress
        :param val_split: share of data to check validation accuracy
        """
        if not self.compiled:
            self.recompile()

        cb = []
        if early_stopping:
            es = EarlyStopping(monitor='val_loss', patience=early_stopping, baseline=baseline)
            cb.append(es)

        if checkpoint:
            if type(checkpoint) == str:
                self.filename = checkpoint
            else:
                os.mkdirs(f"model_selection/best_models/model_{self.model_number}"+self.datasetname)
                self.filename = f"model_selection/best_models/model_{self.model_number}"+self.datasetname+f"/fold_{fold_nr}"+"_weights.{epoch:02d}-{val_loss:.2f}.hdf5"
            mc = ModelCheckpoint(self.filename, monitor='val_loss', save_best_only=True, save_weights_only=True)
            cb.append(mc)
            print(self.filename)

        history = self.model.fit(self.x_train, self.y_train,
                                    batch_size=batch_size,
                                    epochs=epochs,
                                    verbose=verbose,
                                    validation_split=val_split,
                                    callbacks=cb)

        self.train_loss += history.history['loss']
        self.val_loss += history.history['val_loss']
        
        if self.using_min_pred:
            self.set_min_pred()


    def predict(self, x_test=None):
        """ Make prediction on input data

        :param x_test: set an alternative test set
        :param pred_on_train: set to true to predict on the training set
        """
        if not self.compiled:
            print("Model compiles for the first time - weights are initialized...\n")
            self.recompile()
        elif x_test is None:
            x_test = self.x_test

        y_pred = self.model.predict(x_test)
        self.y_pred = y_pred.reshape(y_pred.shape[0])

        if self.using_min_pred:
            mins = np.where(self.y_pred < self.min_pred)[0]
            self.y_pred[mins] = self.min_pred

    def plot_test_points(self, color="black", fig=None, ax=None, density_plot=False, y_true=None, title="Standard"):
        """Plot the latest prediction

        If you set your own dataset in predict() provide a y_true set here.

        :param color: set color of predicted values
        :param fig: Matplotlib figure to plot in
        :param ax: Matplotlib axis to plot on
        :param density_plot: whether to plot the density of the predictions
        use if there are many data points in test set
        :returns: figure
        """
        if self.pred_on_train:
            y_test = self.y_train
        elif y_true is not None:
            y_test = y_true
        else:
            y_test = self.y_test
        y_pred = self.y_pred
        
        if fig is None or ax is None:
            fig, ax = plt.subplots()

        indices = np.argsort(y_test)
        ax.scatter(
            y_test[indices],
            y_pred[indices],
            c=color,
            s=6,
            alpha=0.2
        )
        if density_plot:
            sns.histplot(
                x=y_test[indices],
                y=y_pred[indices],
                bins=20,
                pthresh=.1,
                cmap="mako",
                ax=ax
            )
        ax.plot(
            y_test[indices],
            y_test[indices],
            # marker="o",
            color="firebrick",
        )
        r2 = r2_score(y_test, y_pred)
        ax.legend(title=f"$R^2$: {r2:.3f}", loc="lower right")
        if title == "Standard":
            ax.set_title(f"Predictions for '{self.model.name}'")
        else:
            ax.set_title(title)
        ax.set_xlabel("Actual Expression")
        ax.set_ylabel("Predicted Expression")
        ax.axis('equal')
        ax.set_ylim((y_test[indices[0]]-1,y_test[indices[-1]]+1))

        return fig


    def plot_test_prediction(self, color="black", fig=None, ax=None, y_true=None, title="Standard"):
        """Plot the latest prediction with a two overlapping lines

        If you set your own dataset in predict() provide a y_true set here.

        :param color: set color of predicted values
        :param fig: Matplotlib figure to plot in
        :param ax: Matplotlib axis to plot on
    
        use if there are many data points in test set
        :returns: figure
        """
        if y_true is not None:
            self.y_test = y_true
            y_test = self.y_test
        else:
            y_test = self.y_test
        y_pred = self.y_pred

        r2 = r2_score(y_test, y_pred)
        print("MSE score of model: ", mean_squared_error(y_test, y_pred))
        print("MAE score of model: ", mean_absolute_error(y_test, y_pred))
        print("R2 score of model: ", r2)
        print(f"Normalised covariance between predicted and true is {np.corrcoef(y_test, y_pred)[0, 1]}")

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        indices = np.argsort(y_test)

        x_inds = np.arange(0,y_test.shape[0])
        ax.plot(x_inds, y_test[indices], label="True", color="red")
        ax.scatter(x_inds, y_pred[indices], label="Pred", s=6, color=color, alpha=0.3)
        
        ax.legend(title=f"$R^2$: {r2:.3f}", loc="upper left")
        if title == "Standard":
            ax.set_title(f"Predictions for '{self.model.name}'")
        else:
            ax.set_title(title)
        ax.set_xlabel("Cells sorted by expression")
        ax.set_ylabel("Expression")
        ax.set_ylim((y_test[indices[0]]-1,y_test[indices[-1]]+1))


        return fig

    def plot_train_prediction(self, fig=None, ax=None, title="Standard", color="black"):
        y_pred = self.predict(self.x_train)
        y_train = self.y_train

        r2 = r2_score(y_train, y_pred)
        mse = mean_squared_error(y_train, y_pred)
        msa = mean_absolute_error(y_train, y_pred)

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        indices = np.argsort(y_train)

        x_inds = np.arange(0,y_train.shape[0])
        ax.plot(x_inds, y_train[indices], label="True", color="red")
        ax.scatter(x_inds, y_pred[indices], label="Pred", s=6, color=color, alpha=0.3)
        
        legend_txt = f"$R^2$: {r2:.3f}\nMSE: {mse:.3f}"
        ax.legend(title=f"$R^2$: {r2:.3f}", loc="upper left")
        if title == "Standard":
            ax.set_title(f"Predictions for '{self.model.name}' - trainingset - {self.datasetname}")
        else:
            ax.set_title(title)
        ax.set_xlabel("Cells sorted by expression")
        ax.set_ylabel("Expression")
        ax.set_ylim((y_train[indices[0]]-1, y_train[indices[-1]]+1))


        return fig



    def plot_training(self, fig=None, ax=None, title=None, legend=None, both=True):
        """ Plots training and validation loss over epochs

        :param fig: set a figure to include the plot in
        :param ax: set a position for the plot
        :param title: set title of plot
        :param legend: set legend
        :param both: plot both train and val loss, set to false if trained on 1 epoch
        :return: The figure with the plot
        """

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        ax.plot(self.train_loss)
        if both:
            ax.plot(self.val_loss)

        if title is None:
            ax.set_title(f"Training loss for {self.model.name}")
        else:
            ax.set_title(title)
        if legend is None:
            ax.legend(['train', 'validation'], loc='upper right')
        else:
            ax.legend(legend, loc='upper right')
        ax.set_ylabel('loss')
        ax.set_xlabel('batch')

        return fig

    @property
    def r2(self):
        return r2_score(self.y_test, self.y_pred)

    @property
    def mse(self):
        return mean_squared_error(self.y_test, self.y_pred)

    @property
    def mae(self):
        return mean_absolute_error(self.y_test, self.y_pred)
    
    @property
    def epochs_completed(self):
        return len(self.train_loss)


class Histories(Callback):
    """Records val_loss when training for only one epoch
    set callback = [instance of histories] to record loss"""
    def on_train_begin(self, logs={}):
        self.train_losses = []

    def on_batch_end(self, batch, logs={}):
        self.train_losses.append(logs.get('loss'))


def model_constructor(layers, activations, input_shape):

    optimizer = ko.Adam()
    model = km.Sequential(name="Dense_model")
    for i in range(len(layers)-1):
        model.add(kl.Dense(layers[i], activation=activations[i]))
    model.add(kl.Dense(layers[i+1], activation="linear", name="Output"))
    model(tf.ones((1, input_shape)))
    model.compile(optimizer=optimizer, loss="mse")
    return model


def save_object(object_, path_to_file, overwrite=False):
    if path.exists(path_to_file):
        if overwrite:
            with open(path_to_file, 'wb') as f:
                pickle.dump(object_, f)
        else:
            print("File already exists. Use overwrite=True to overwrite it.")
    else:
        with open(path_to_file, 'wb') as f:
                pickle.dump(object_, f)

def load_object(path_to_file):
    try:
        with open(path_to_file, "rb") as f:
            object_ = pickle.load(f)
        return object_
    except FileNotFoundError:
        print("File does not exist. You probably have a typo.")
    



def run_multiple_models(models, activations, df, name_of_y, datasetname, kfold=5, normalization="standard"):
    """Test multiple model configurations on dataset with K-fold cross validation

        Output is put in a file called *datasetname*_model_info.txt
        And the predictions of the last model for each cv is saved in figures/
        
    
    Parameters
    ----------
    models : list of lists of ints
        A list of lists where each element is the number of nodes in a layer. 

    activations : list of list of strings
        Each element is the name of the activation for the corresponding layer 
        in the models sublists. 
    
    df : pandas.DataFrame
        The data to be trained and tested on.

    name_of_y : string
        Name of the column that is used as output. 
    
    datasetname : string
        Name of the dataset for nice formatting in the output files.
    
    kfold : int
        the number of folds in k-fold cross validation.
    
    normalization : string
        Normalizaton method to be used - options: 'standard', 'minmax', 'none'
    """

    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # to remove tf infomessages

    
    # Data handling
    print("Computing correlation...")
    #df = correlation_reduction(df, limit=0.05, target=name_of_y)

    print("Normalizing...")
    X = df.drop([name_of_y], axis=1)
    Y = df[name_of_y]

    # Kfold Splitter
    cv = KFold(n_splits = kfold)


    # Storing results
    results = np.zeros((len(models), 3))

    results_df = np.zeros((len(models), 3*kfold+3))

    # Stores in txt file
    infofilename = 'model_selection/'+datasetname+'_model_info.txt'

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    with open(infofilename, 'a') as f:
        f.write(f"\n\n\n\n\n\n\nNEW SESSION - {current_time} \n")
        f.write(f"\nNumber of models = {len(models)}\n\n")
    
    print("Constructing models...")
    for i, model_struct in enumerate(models):
        for activation in activations:
            kf_cv_results = np.zeros((kfold, 3))
            epochs = 0
            print("Running model: ", models[i])
            splits = cv.split(X)
            for j, [train_ind, test_ind] in enumerate(splits):
                print(f"Running fold {j+1}/{kfold}")
                X_train = X.iloc[train_ind]
                y_train = Y[train_ind]
                X_test = X.iloc[test_ind]
                y_test = Y[test_ind]

                model = model_constructor(model_struct, activation, X_train.shape[1])
                net = FFNN(model, "adam", model_number=str(i))
                net.recompile()
                net.set_eta(0.0005)
                net.set_trainingset(X_train, y_train, datasetname+activation[0])
                net.set_testingset(X_test, y_test)
                net.train(batch_size=32, epochs=30, early_stopping=6, verbose=0, checkpoint=True, fold_nr=j)
                #net.model.load_weights(checkpoint_filepath)
                kf_cv_results[j][0] = net.mse
                kf_cv_results[j][1] = net.mae
                kf_cv_results[j][2] = net.r2
                epochs += net.epochs_completed

                results_df[i][j] = net.mse
                results_df[i][1+kfold+j] = net.mae
                results_df[i][2+2*kfold+j] = net.r2

                if j == (kfold-1):
                    fig, ax = plt.subplots(figsize=(6,6), dpi=300)
                    title = f"Model {i+1} - {activation[0]} - {datasetname}"
                    net.plot_test_line(fig=fig, ax = ax, title = title)
                    fig_filename = "model_selection/figures/"+ f"Model_{i}_{activation[0]}_{datasetname}" + ".png"
                    fig.savefig(fig_filename)
            
            results_df[i][kfold] = results_df[i][:kfold].mean()
            results_df[i][1+2*kfold] = results_df[i][kfold:2*kfold].mean()
            results_df[i][-1] = results_df[i][2*kfold:-1].mean()
        

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            with open(infofilename, 'a') as f:
                f.write(f"     {current_time}\n     Model {i+1}: ")
                f.write(activation[0])
                f.write(f" {model_struct} - finished with avg epochs: {epochs/kfold}\n")
                f.write(f"                avg MSE: {results[i][0]:.5f}\n")   
                f.write(f"                         {list(kf_cv_results[:,0])}\n") 
                f.write(f"                         max: {np.amax(kf_cv_results[:,0]):.5f}   min: {np.amin(kf_cv_results[:,0]):.5f}\n")
                f.write(f"                avg mae: {results[i][1]:.5f}\n")
                f.write(f"                         {list(kf_cv_results[:,1])}\n") 
                f.write(f"                         max: {np.amax(kf_cv_results[:,1]):.5f}   min: {np.amin(kf_cv_results[:,1]):.5f}\n")
                f.write(f"                 avg R2: {results[i][2]:.5f}\n")
                f.write(f"                         max: {np.amax(kf_cv_results[:,2]):.5f}   min: {np.amin(kf_cv_results[:,2]):.5f}\n")
                f.write(f"                         {list(kf_cv_results[:,2])}\n") 
                f.write(f"")
    
    columns = []
    for i in range(kfold):
        columns.append("MSE_"+str(i))
    columns.append("Average MSE")
    for i in range(kfold):
        columns.append("MAE_"+str(i))
    columns.append("Average MAE")
    for i in range(kfold):
        columns.append("R2_"+str(i))
    columns.append("Average R2")

    index = []
    for model in models:
        index.append(str(model))

    results_df = pd.DataFrame(results_df, index=index, columns=columns)
    results_df.index.name = "Models"

    return results_df




if __name__ == "__main__": 

    cancer = True
    myeloid = False

    model_1 = [32, 16, 1]
    model_2 = [16, 8, 1]
    model_4 = [16, 1]
    model_5 = [32, 1]
    activation_1 = ["relu", "relu", "linear"]
    activation_2 = ["sigmoid", "sigmoid", "linear"]

    models = [model_1, model_2, model_4, model_5]
    activations = [activation_1]

    # Myeloid:

    if myeloid:

        df = pd.read_pickle("/data/severs/rna_myeloid.pkl")

        results_myeloid = run_multiple_models(models, activations, df, "ESR1", "myeloid", kfold=5)

        with open("model_selection/results_myeloid.pkl", 'wb') as f:
            pickle.dump(results_myeloid, f)


    # Cancer
    if cancer:
        
        df = pd.read_pickle("/data/severs/rna_scaled_cancer.pkl")

        df = transpose_set_ind(df)

        results_df_cancer = run_multiple_models(models, activations, df, "ESR1", "cancer", kfold=8, normalization="standard")

        
        with open("model_selection/results_df_cancer.pkl", 'wb') as f:
            pickle.dump(results_df_cancer, f)
