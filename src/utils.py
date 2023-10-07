import numpy as np
import scipy.stats as stats
import pandas as pd
import seaborn as sns
from typing import List, Tuple
from matplotlib import pyplot as plt
import itertools
from sklearn.metrics import (roc_curve, roc_auc_score)

"""cramers_v
Computes Cramer's V statistic (with bias correction) for categorical-categorical association.
Args:
    df (DataFrame): input dataframe
    var1 (str): name of the first categorical variable
    var2 (str): name of the second categorical variable
     bias_correction (bool): whether to apply bias correction
Returns:
    float: Cramer's V statistic for categorical-categorical association
"""
def cramers_v(df:pd.DataFrame, var1:str, var2:str, bias_correction:bool=True)->float:
    df = df.copy()
    
    # crosstab of the two variables
    crosstab =np.array(pd.crosstab(df[var1],df[var2], rownames=None, colnames=None))
    
    # chi2 (chi-sq statistics of the two variables)
    chi2, p, dof, ex = stats.chi2_contingency(crosstab)
    
    # Total number of observations
    n = np.sum(crosstab)
    
    # Take the min(# of rows -1, # of cols -1)
    nrow = crosstab.shape[0]
    ncol = crosstab.shape[1]

    mini = min(nrow-1, ncol-1)
    
    # Let phi be chi-sq test statistics/N
    phi = chi2/n
    
    # Compute Cramer's V
    V = np.sqrt(phi/mini)
    
    if bias_correction:
        # Adjust phi
        phi_adj = max(0, (phi - ((ncol-1)*(nrow-1)/(n-1))))
        
        # ncol adjusted
        ncol_adj = ncol-((ncol-1)**2/(n-1))
        # nrow adjusted
        nrow_adj = nrow-((nrow-1)**2/(n-1))
        
        # mini adjusted
        mini_adj = min(nrow_adj-1, ncol_adj-1)
        
        # Cramer's V with bias correction
        V_adj = np.sqrt(phi_adj/mini_adj)
        
        return V_adj
    else:
        return V
    
    

    """cat_cat_stackplot
    Args:
        df (DataFrame): input dataframe
        x (str): name of the first categorical variable
        y (str): name of the second categorical variable
    Returns:
        None: Stacked bar plot
    """    
def cat_cat_stackplot(df:pd.DataFrame, x:str, y:str):
    data = df.copy()
    
    # Compute the percentage of each level of x for each y group
    counts = (data
          .groupby(y)[x]
          .value_counts(normalize=True)
          .mul(100)
          .reset_index(name="percent"))
    
    # I need to convert this dataframe from long to wide
    counts_wide = counts.pivot(index = y, columns = x, values= "percent")
    counts_wide=counts_wide.reset_index()
    
    # Stack bar plot
    
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(2,2,1) 
    counts_wide.plot(x=y, 
                     kind='barh', 
                     stacked=True,
                     ax = ax1,
                     title=f"Stacked Bar Graph: {y} X {x}")

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
    
    
    
    
def get_num_cat_vars(df: pd.DataFrame) -> Tuple[List, List]:
    """
    get_num_cat_vars:
    Given an input dataframe and a categorical variable,
    one-hot encode the variable, drop the original variable,
    and merge the one-hot encoded columns to the
    input dataframe.

    Args:
        df (pd.DataFrame): input dataframe
    Returns:
        cat_vars(List): a list of categorical variable names
        num_vars(List): a list of numeric variable names
    """

    # Make a copy of the input dataframe
    data = df.copy()

    # Get categorical variables in the dataframe
    cat_vars = [var for var in data.columns if data[var].dtype == "O"]

    # Get numeric variables in the dataframe (even those that are integers but
    # are actually ordinal)
    num_vars = data.select_dtypes(include=np.number).columns.values.tolist()

    return cat_vars, num_vars



def plot_confusion_matrix(cm, 
                          classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`
    
    Args:
        cm (array): confusion matrix
        classes (list): a list of class names
        normalize (bool): whether to normalize the confusion matrix
        title (str): title of the plot
        cmap (matplotlib colormap): colormap
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)
    
    
    
def plot_multiclass_roc(y_val, y_scores, mapping, figsize=(20, 8)):
    # structures
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # get dummy variables for y_val, one for each level
    y_val_dummies = pd.get_dummies(y_val, drop_first=False).values
    
    # Compute fpr and tpr for each class
    # Also compute the ROC_AUC for each class
    for k,v in mapping.items():
        fpr[v], tpr[v], _ = roc_curve(y_val_dummies[:, k-1], y_scores[:, k-1])
        roc_auc[v] = roc_auc_score(y_val_dummies[:, k-1], y_scores[:, k-1])

    # plot roc for each class
    fig, ax = plt.subplots(figsize=figsize)
    plt.style.use("fivethirtyeight")
    plt.rcParams["font.size"] = 12
    
    for _,v in mapping.items():
        ax.plot(fpr[v], tpr[v], label=f"ROC curve for class {v} (area = {round(roc_auc[v],2)})")
    
    # plot settings
    ax.plot([0, 1], [0, 1], 'k--') # plot the 45 deg line
    
    ax.set_xlim([0.0, 1.0]) # set x and y limits
    ax.set_ylim([0.0, 1.05])
    
    ax.set_xlabel('False Positive Rate') # set x and y labels and title
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROCs')
    
    ax.legend(loc="best")
    plt.show()
    
    

