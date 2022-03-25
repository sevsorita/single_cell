from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score

#from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import scipy

import xgboost as xgb
import shap


def add_zeros(s, width=2):
    s = str(s)
    while len(s) < width:
        s = "0" + s
    return s


def get_index(gene : str, X: pd.DataFrame):
    return np.where(X.columns==gene)[0][0]


def z_score_comparison(a, b, p=True):
    n_a = a.shape[0]
    n_b = b.shape[0]
    z = (a.mean() - b.mean())/(np.sqrt(a.std()/np.sqrt(n_a) + b.std()/np.sqrt(n_b)))
    if p:
        return scipy.stats.norm.sf(abs(z))
    return z


def root_mse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())


def train_test_val_split(X, y, test_size=0.2, val_size = 0.2, random_state=None):
    X_red, X_test, y_red, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    val_size = val_size/(1-test_size)
    X_train, X_val, y_train, y_val = train_test_split(X_red, y_red, test_size=val_size, random_state=random_state)

    return X_train, X_test, X_val, y_train, y_test, y_val

def plot_prediction_alt(y_true, y_pred, fig=None, ax=None, title="", **kwargs):
    r2 = r2_score(y_true, y_pred)
    rmse = root_mse(y_true, y_pred)

    sorter = y_true.argsort()

    if fig is None:
        fig, ax = plt.subplots()
    if "color" not in kwargs:
        kwargs["color"] = "black"
    ax.set_title(title)
    ax.plot((y_true[sorter].max(),y_true[sorter].min()), (y_true[sorter].max(),y_true[sorter].min()), label="True", color="red")
    ax.scatter(y_true[sorter], y_pred[sorter], s=6, alpha=0.5, label="Prediction", **kwargs)
    ax.legend(title = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}")
    ax.set_ylabel("Predicted gene expression")
    ax.set_xlabel("True gene expression")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    return fig

def plot_prediction(y_true, y_pred, fig=None, ax=None, title="", **kwargs):
       
    r2 = r2_score(y_true, y_pred)
    rmse = root_mse(y_true, y_pred)

    sorter = y_true.argsort()

    if fig is None:
        fig, ax = plt.subplots()
    
    if "color" not in kwargs and "c" not in kwargs:
        kwargs["color"] = "black"
    elif 'c' in kwargs:
        kwargs["c"] = kwargs["c"][sorter]
    
    
    ax.set_title(title)
    
    if "cmap" in kwargs and (kwargs["cmap"] == "hsv" or kwargs["cmap"] =="nipy_spectral"):
        true_color="black"
    else:
        true_color="red"
    
    ax.plot(np.arange(y_true.shape[0]), y_true[sorter], label="True", color=true_color)
    sc = ax.scatter(np.arange(y_true.shape[0]), y_pred[sorter], s=6, alpha=0.5, label="Prediction", **kwargs)
    ax.set_ylabel("ESR1 expression")
    ax.set_xlabel("Cells sorted by target gene expression")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    legend1 = ax.legend(*sc.legend_elements(),
                    bbox_to_anchor=(1, 1), title="Patients")
    
    if 'c' in kwargs:
        for i, p in enumerate(np.arange(kwargs["c"].min(),kwargs["c"].max()+1)):
            
            if p in kwargs["c"]:
                r2 = r2_score(y_true[(kwargs["c"]==p)], y_pred[(kwargs["c"]==p)])
                if r2 != float("nan"):
                    n = sum(kwargs["c"]==p)
                    legend1.get_texts()[i].set_text(f"{p} - R2:{r2:.2f}, n={n}")
                    
            else:
                legend1.get_texts()[i].set_text(f"{p} - n=0")
            
            #print(str(legend1.get_texts()[i]).split("{")[-1].split("}")[0])
            #legend1.get_texts()[i].set_text(f"{t} {i}")
    
    ax.add_artist(legend1)
    ax.legend(title = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}")
    return fig

def plot_classification(y_true, y_pred, classifier="binary", fig=None, ax=None, title=""):
    accuracy = accuracy_score(y_true, y_pred)
    F1 = f1_score(y_true, y_pred)

def metric_converter(metrics):
    """Convert list of string metrics to function metrics"""

    output = []

    for metric in metrics:
        if metric == "mse":
            output.append(mean_squared_error)
        elif metric == "r2": 
            output.append(r2_score)
        elif metric == "mae": 
            output.append(mean_absolute_error)
        elif metric == "rmse":
            output.append(root_mse)
        else:
            raise ValueError(f"{metric} not accepted. Try 'rmse', 'mse', 'r2', or 'mae'")
            
    return output


def KerasCV(model_constructor, X, y, *, folds=5, epochs=20, batch_size=32, patience = 8, random_state=None, metrics = ["r2","rmse"]):
    

    cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    splits = cv.split(X,y)

    results = np.zeros((folds, len(metrics)))

    filename = "cv_best_model.hdf5"

    metric_funcs = metric_converter(metrics)

    sc = StandardScaler()

    for i, [train_inds, test_inds] in enumerate(splits):
        X_train, y_train = X.iloc[train_inds], y[train_inds]
        X_test, y_test = X.iloc[test_inds], y[test_inds]
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        model = model_constructor()
        #mc = ModelCheckpoint(filename, monitor='val_loss', save_best_only=True, save_weights_only=True)
        #es = EarlyStopping(monitor='val_loss', patience=8)
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

        model.load_weights(filename)
        y_pred = model.predict(X_test)
        y_pred = y_pred.reshape(y_pred.shape[0])
        for j, metric in enumerate(metric_funcs):
            results[i, j] = metric(y_pred, y_test)
    
    results = pd.DataFrame(results.mean(axis=0).reshape((1,len(metrics))))
    
    results.columns = metrics

    return results


def XGB_CV(model_constructor, X, y, folds=5, random_state=None, metrics = ["r2","rmse"]):
    cv = KFold(n_splits=folds, shuffle=True, random_state=random_state)

    splits = cv.split(X,y)

    results = np.zeros((folds, len(metrics)))

    metric_funcs = metric_converter(metrics)

    for i, [train_inds, test_inds] in enumerate(splits):
        print(f"Fold {i+1}/{folds}\r", end="")
        X_train, y_train = X.iloc[train_inds], y[train_inds]
        X_test, y_test = X.iloc[test_inds], y[test_inds]

        model = model_constructor()
        
        model.fit(X_train, y_train,verbose=False)

        y_pred = model.predict(X_test)

        for j, metric in enumerate(metric_funcs):
            results[i, j] = metric(y_test, y_pred)
    
    # results = pd.DataFrame(results.mean(axis=0).reshape((1,len(metrics))))
    
    # results.columns = metrics

    return results

def get_biggest_shap(shap_values, X):
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    sorter = np.argsort(mean_abs_shap)
    biggest_inds = sorter[-9:]
    
    biggest_exp = []
    values = np.zeros(9)
    for i, ind in enumerate(biggest_inds):
        biggest_exp.append(X.iloc[:,ind].name)
        values[i] = mean_abs_shap[ind]
    
    values = np.flip(values)
    biggest_exp.reverse()
    return biggest_exp, values


def shap_tester(X, y, iters = 100):
    
    r2 = np.zeros(iters)
    rmse = np.zeros(iters)

    shap_dict = {}

    all_shap = np.zeros((iters, X.shape[1]))

    for i in range(iters):

        X_train, X_test, X_val, y_train, y_test, y_val = train_test_val_split(X, y)

        eval_set = [(X_test, y_test)]

        model = xgb.XGBRegressor(objective="reg:squarederror", eta=0.03, n_estimators=1000, n_jobs=80, max_depth=3)

        model.fit(X_train, y_train,verbose=False, early_stopping_rounds = 10, eval_set = eval_set)

        y_pred = model.predict(X_val)

        r2[i] = r2_score(y_val, y_pred)
        rmse[i] = root_mse(y_val, y_pred)

        explainer = shap.Explainer(model)
        shap_values = explainer(X_val)
        
        all_shap[i] = np.abs(shap_values.values).mean(axis=0)

    mean_shap = np.abs(all_shap).mean(axis=0)

    std_shap = np.abs(all_shap).std(axis=0)

    df = pd.DataFrame(mean_shap, columns=["mean"])
    df["std"] = std_shap
    df["genes"] = list(X.columns)
    df.set_index("genes", inplace=True)
    df.sort_values("mean", axis=0, ascending=False, inplace=True)
    
    accuracy = {"avg_r2": r2.mean(), "avg_rmse": rmse.mean(), "r2_std":r2.std(), "rmse_std":rmse.std()}

    return accuracy, df


def convert_dict_to_df(shap_results):
    genes = []
    means = []
    stds = []

    for key in shap_results.keys():
        genes.append(key)
        means.append(shap_results[key]["mean"])
        stds.append(shap_results[key]["std"])
    shap_results_df = pd.DataFrame(genes)
    shap_results_df["mean"] = means
    shap_results_df["std"] = stds
    shap_results_df.columns = ["gene", "mean", "std"]
    shap_results_df= shap_results_df.set_index(["gene"])
    shap_results_df.sort_values("mean", axis=0, ascending=True, inplace=True)
    return shap_results_df


def plot_importance(df_, n_bars=20,  fig=None, ax=None, title=""):
    
    df  = df_.iloc[:n_bars].sort_values("mean", axis=0, ascending=True, inplace=False)
    
    genes = list(df.index)

    y_pos = np.arange(len(genes))
    SHAP = list(df["mean"])
    error = list(df["std"])    
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 6), dpi=100)
    ax.barh(y_pos, SHAP, xerr=error, align='center', color="tomato", zorder=10)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(genes)
    
    ax.set_xlabel("mean(|SHAP value|)")

    ax.set_title(title)
    
    for i, v in enumerate(SHAP):
        if error[i] == 0:
            err = error[i-1]
        else:
            err = error[i]
        ax.text(v + err+0.001, i-0.15, f"{v:.3f}", color='tomato')
        
    plt.tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are off
        labelleft=True) # labels along the bottom edge are off

    ax.set_xticks(np.arange(0, 0.041, 0.01))
    
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    return fig, ax



def plot_training_xgb(evaluation_results, fig=None, ax=None, title=""):
    """Plot the training curve of a model
    

    """
    # Evaluation results
    for key in evaluation_results['validation_0']:
        scoring = key

    if fig is None:
        fig, ax = plt.subplots(dpi=100)

    err_train = evaluation_results['validation_0'][scoring] # Train ‘error’ metric
    
    ax.plot(err_train)
    
    if 'validation_1' in evaluation_results.keys():
        err_test = evaluation_results['validation_1'][scoring] # Test ‘error’ metric
    
        if scoring == "auc":
            min_x = np.argmax(evaluation_results['validation_1'][scoring])
            min_y_1 = np.amax(evaluation_results['validation_0'][scoring])
            min_y_2 = np.amin(evaluation_results['validation_1'][scoring])
            s = "Max"
        else:
            min_x = np.argmin(evaluation_results['validation_1'][scoring])
            min_y_1 = np.amin(evaluation_results['validation_0'][scoring])
            min_y_2 = np.amax(evaluation_results['validation_1'][scoring])
            s = "Min"

        ax.plot(err_test)
        ax.plot((min_x, min_x), (min_y_1, min_y_2), "r--")
    
    
    ax.set_xlabel('Nbr of trees (n_estimators)')
    ax.set_ylabel(f'{scoring}')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    if 'validation_1' in evaluation_results.keys():
        ax.legend(['Train', 'Test', f"{s}: {evaluation_results['validation_1'][scoring][min_x]:.4f}"])
    else:
        ax.legend(['Train', 'Test'])
    ax.set_title(title)
    return fig, ax





import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import itertools



def plot_confusion_matrix(cm, labels=False, title='Confusion matrix', cmap=plt.cm.Oranges, sum_stats=True, fig=None, ax=None):
    """Make a plot of a sklearn confucion matrix
    
    Parameters: 
    -----------
    cm : sklearn confucion matrix
    labels : list of str
    """

    # Calculate scores
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cm) / float(np.sum(cm))

        #if it is a binary confusion matrix, show some more stats
        if len(cm)==2:
            #Metrics for Binary Confusion Matrices
            precision = cm[1,1] / sum(cm[:,1])
            recall    = cm[1,1] / sum(cm[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""

    if fig is None and ax is None:
         ax = plt

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)

    tick_marks = np.arange(len(cm))
    if labels:
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(labels, rotation=30)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(labels)
    thresh = cm.max() / 2.
    if cm.max()>1:
        decimals = ".0f"
    else:
        decimals = ".3f"
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i,  format(cm[i, j], decimals),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label' + stats_text)



def make_confusion_matrix(cf,
                          group_names=None,
                          categories='auto',
                          count=True,
                          percent=True,
                          cbar=True,
                          xyticks=True,
                          xyplotlabels=True,
                          sum_stats=True,
                          figsize=None,
                          cmap='Blues',
                          title=None, 
                          fig=None, 
                          ax=None):
    '''
    This function will make a pretty plot of an sklearn Confusion Matrix cm using a Seaborn heatmap visualization.

    Arguments
    ---------
    cf:            confusion matrix to be passed in

    group_names:   List of strings that represent the labels row by row to be shown in each square.

    categories:    List of strings containing the categories to be displayed on the x,y axis. Default is 'auto'

    count:         If True, show the raw number in the confusion matrix. Default is True.

    normalize:     If True, show the proportions for each category. Default is True.

    cbar:          If True, show the color bar. The cbar values are based off the values in the confusion matrix.
                   Default is True.

    xyticks:       If True, show x and y ticks. Default is True.

    xyplotlabels:  If True, show 'True Label' and 'Predicted Label' on the figure. Default is True.

    sum_stats:     If True, display summary statistics below the figure. Default is True.

    figsize:       Tuple representing the figure size. Default will be the matplotlib rcParams value.

    cmap:          Colormap of the values displayed from matplotlib.pyplot.cm. Default is 'Blues'
                   See http://matplotlib.org/examples/color/colormaps_reference.html
                   
    title:         Title for the heatmap. Default is None.

    '''


    # CODE TO GENERATE TEXT INSIDE EACH SQUARE
    blanks = ['' for i in range(cf.size)]

    if group_names and len(group_names)==cf.size:
        group_labels = ["{}\n".format(value) for value in group_names]
    else:
        group_labels = blanks

    if count:
        group_counts = ["{0:0.0f}\n".format(value) for value in cf.flatten()]
    else:
        group_counts = blanks

    if percent:
        group_percentages = ["{0:.2%}".format(value) for value in cf.flatten()/np.sum(cf)]
    else:
        group_percentages = blanks

    box_labels = [f"{v1}{v2}{v3}".strip() for v1, v2, v3 in zip(group_labels,group_counts,group_percentages)]
    box_labels = np.asarray(box_labels).reshape(cf.shape[0],cf.shape[1])


    # CODE TO GENERATE SUMMARY STATISTICS & TEXT FOR SUMMARY STATS
    if sum_stats:
        #Accuracy is sum of diagonal divided by total observations
        accuracy  = np.trace(cf) / float(np.sum(cf))

        #if it is a binary confusion matrix, show some more stats
        if len(cf)==2:
            #Metrics for Binary Confusion Matrices
            precision = cf[1,1] / sum(cf[:,1])
            recall    = cf[1,1] / sum(cf[1,:])
            f1_score  = 2*precision*recall / (precision + recall)
            stats_text = "\n\nAccuracy={:0.3f}\nPrecision={:0.3f}\nRecall={:0.3f}\nF1 Score={:0.3f}".format(
                accuracy,precision,recall,f1_score)
        else:
            stats_text = "\n\nAccuracy={:0.3f}".format(accuracy)
    else:
        stats_text = ""


    # SET FIGURE PARAMETERS ACCORDING TO OTHER ARGUMENTS
    if figsize==None:
        #Get default figure size if not set
        figsize = plt.rcParams.get('figure.figsize')

    if xyticks==False:
        #Do not show categories if xyticks is False
        categories=False


    # MAKE THE HEATMAP VISUALIZATION
    if fig and ax is None:
        fig, ax = plt.subplot(figsize=figsize)
    
    sns.heatmap(cf,annot=box_labels,fmt="",cmap=cmap,cbar=cbar,xticklabels=categories,yticklabels=categories)

    if xyplotlabels:
        plt.ylabel('True label')
        plt.xlabel('Predicted label' + stats_text)
    else:
        plt.xlabel(stats_text)


def shap_bar(shap_vals, X, dpi=80, others=False):
    abs_shap = np.abs(shap_vals).mean(axis=0)
    sorter = np.argsort(abs_shap)
    fig, ax = plt.subplots(figsize=(4,6), dpi=dpi)

    bar_vals = abs_shap[sorter][-20:]
    bar_pos = np.linspace(0,40,20)
    yticks = list(X.columns[sorter][-20:])
    if others:
        bar_vals[0] = abs_shap[sorter][:-21].sum()
        yticks[0] = "Sum of all others"
    ax.barh(bar_pos, bar_vals, align='center', height=1, color="#FF0051")
    ax.set_yticks(bar_pos)
    ax.set_yticklabels(yticks)
    ax.set_xlabel("Average absolute SHAP value")
    ax.set_title('The models highest "coefficients"')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i, v in enumerate(bar_vals):
        ax.text(v + 0.001, bar_pos[i]-0.55, f"+{v:.3f}", color="#FF0051")
    plt.show()


def shap_corr(shap_vals, X, corr_df, dpi=80, title=""):
    abs_shap = np.abs(shap_vals).mean(axis=0)
    sorter = np.argsort(abs_shap)
    genes = X.columns[sorter][-20:]
    corr = corr_df.loc[genes]
    fig, ax = plt.subplots(figsize=(4,6), dpi=dpi)
    x_pos = corr_df.correlation.abs().sort_values()[-20]
    barh_plot(corr, dpi=dpi, title=title, fig=fig, ax=ax)

def barh_plot(corr_df, dpi=80, title="", fig=None, ax=None):
    
    n = corr_df.shape[0]
    if fig==None and ax==None:
        fig, ax = plt.subplots(figsize=(n/5,n*0.3), dpi=dpi)
    abs_corr = corr_df["correlation"].abs().values
    color = (corr_df["correlation"] == corr_df["correlation"].abs()).values
    bars = ax.barh(np.linspace(0,n*2,n), abs_corr, align='center', height=1, color="#FF0051")
    negative_corr = False
    for i, c in enumerate(color):
        if not c:
            bars[i].set_color("#008BFB")
            negative_corr = True
    ax.set_yticks(np.linspace(0,n*2,n))
    ax.set_yticklabels(corr_df.index)
    ax.set_xlabel("|Correlation|")
    ax.set_ylabel("Gene")
    ax.set_title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    for i, v in enumerate(abs_corr):
            ax.text(v + 0.005, np.linspace(0,n*2,n)[i]-0.55, f"{corr_df.correlation.values[i]:.3f}")

    if negative_corr:
        import matplotlib.patches as mpatches
        red_patch = mpatches.Patch(color='#FF0051', label='Positive correlation')
        blue_patch = mpatches.Patch(color='#008BFB', label='Negative correlation')
        plt.legend(handles=[red_patch, blue_patch], bbox_to_anchor=(1.1, 1), loc='upper left')

    #plt.legend()

    plt.show()  


import typing
from typing import List

class Data:
    
    def __init__(self, X: pd.DataFrame, y: pd.Series, train_inds: List=None, test_inds: List=None, val_inds: List=None, weights = None) -> None:
        self.X = X
        self.y = y

        self.split_data(train_inds, test_inds, val_inds)
        self.set_weights(weights)

    def split_data(self, train_inds: List[int]=None, test_inds: List[int]=None, val_inds: List[int]=None):
        self.train_inds=train_inds
        self.test_inds=test_inds
        self.val_inds=val_inds

        if type(train_inds) is List:
            if type(test_inds) is List:
                if len(train_inds) + len(test_inds) != len(set(train_inds+set(test_inds))):
                    raise ValueError("There are overlapping indices between training and testset")

    def set_weights(self, weights):
        try:
            if len(weights) != len(self.y):
                raise ValueError("Length of weights must match dataset")
            self.weights = weights
        except TypeError:
            pass 

    @property
    def X_train(self):
        if type(self.train_inds) is list:
            return self.X.iloc[self.train_inds]
        else:
            raise IndexError("train_inds are not set. Set them through the split_data function")

    @property
    def y_train(self):
        return self.y[self.train_inds]

    @property
    def X_test(self):
        return self.X.iloc[self.test_inds]

    @property
    def y_test(self):
        return self.y[self.test_inds]

    @property
    def weights_train(self):
        return self.weights[self.train_inds]

    
    


