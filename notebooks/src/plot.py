import matplotlib.pyplot as plt
import numpy as np

from src.modules import r2_score, root_mse


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
        pns = kwargs["c"]
        for i, p in enumerate(np.arange(pns.min(),pns.max()+1)):
            print(p, i)
            if p in pns.values:
                r2 = r2_score(y_true[(pns==p)], y_pred[(pns==p)])
                if r2 != float("nan"):
                    n = sum(pns==p)
                    legend1.get_texts()[i].set_text(f"{p} - R2:{r2:.2f}, n={n}")
                    
            else:
                legend1.get_texts()[i].set_text(f"{p} - n=0")
            
    
    ax.add_artist(legend1)
    ax.legend(title = f"$R^2$ = {r2:.3f}\nRMSE = {rmse:.3f}")
    return fig


def plot_prediction(y_true, y_pred, fig=None, ax=None, title="", sort=True, show_legend=True, **kwargs):
    """ Plot the prediction vs ground truth. If "c" is passed as a kwarg the predictions
        will be colored by cycle.

    Arguments:
        y_true : np.ndarray of the true values
        y_pred : np.ndarray of the predictions
        fig : a matplotlib figure, pass it to customize the plot further
        ax : a matplotlib subplot ax, pass it to customize the plot further,
             must be passed with fig
        title : string, the title of the plot
        sort : sort y_true and y_pred from smallest to biggest true value, good
               for messy data
    Optional but recommmended additonal kwarg:
        c: list, np.ndarray of ints, specify the group that each sample belongs to

    Returns plot
    """

    r2_tot = r2_score(y_true, y_pred)
    rmse_all = root_mse(y_true, y_pred)

    if sort:
        sorter = y_true.argsort()
    else:
        sorter = np.arange(y_true.shape[0])

    if fig is None:
        fig, ax = plt.subplots()
    
    if "color" not in kwargs and "c" not in kwargs:
        kwargs["color"] = "black"
    elif "c" in kwargs:
        kwargs["c"] = kwargs["c"][sorter]
    
    ax.set_title(title)
    
    sc = ax.scatter(np.arange(y_true.shape[0]), y_pred[sorter], s=6, alpha=0.5, label="Prediction",zorder=2, **kwargs)
    ax.plot(np.arange(y_true.shape[0]), y_true[sorter], label="True", color="red", zorder=1)
    ax.set_ylabel("Expression")
    if sort:
      ax.set_xlabel("<- lower, higher ->")
    else:
      ax.set_xlabel("sample number")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # If colo
    if 'c' in kwargs and show_legend:
        legend1 = ax.legend(*sc.legend_elements(),
                    bbox_to_anchor=(1, 1), title="Results per patient \n               RMSE   n")
        pns = kwargs["c"]
        for i, p in enumerate(np.arange(pns.min(),pns.max()+1)):
            if p in pns.values:
                rmse = root_mse(y_true[(pns==p)], y_pred[(pns==p)])
                n = sum(pns==p)
                legend1.get_texts()[i].set_text(f"{p} - {rmse:2.2f} |{n:3.0f}")
            else:
                legend1.get_texts()[i].set_text(f"{p} -  ~    |")
        ax.add_artist(legend1)
    ax.legend(title = f"$R^2$ = {r2_tot:.3f}\nRMSE = {rmse_all:.3f}")
    return fig
