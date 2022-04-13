import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def pairplot(
    data: pd.DataFrame, 
    hue: str = 'target', 
    vars: list =[], 
    show: bool = True,
    save: str = None,
    ):
    """Pair plot.
    
    Args:
        data: a pandas DataFrame
        hue: target
        vars: data variables to be used.
        show: show the plotted window?
        save: save name for the figure
    
    """
    sns.pairplot(
        data,
        hue = hue,
        vars = vars
    )
    if save is not None:
        plt.savefig(save)

    if show:
        plt.show()
