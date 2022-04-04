import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def pairplot(
    data: pd.DataFrame, 
    hue: str = 'target', 
    vars: list =[], 
    show: bool = True
    ):
    """Pair plot.
    
    Args:
        data: a pandas DataFrame
        hue: target
        vars: data variables to be used.
        show: show the plotted window?
    
    """
    sns.pairplot(
        data,
        hue = hue,
        vars = vars
    )
    if show:
        plt.show()
