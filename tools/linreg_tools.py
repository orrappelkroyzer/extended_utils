
import pandas as pd
from sklearn import preprocessing
import statsmodels.api as sm
from statsmodels.sandbox.regression.gmm import IV2SLS
import numpy as np
from scipy import stats
import os, sys

from pathlib import Path
local_python_path = str(Path(__file__).parents[2])
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json") 

def linreg(df,
           independent_variables,
           dependent_variables,
           categoric_independent_variables=[],
           weights_variable=None,
           normalizeX=True, 
           normalizeY=True,
           nesting_variable=None,
           df_with_const=False,
           **kw_args):
    
    X_regular = [df[independent_variables]]
    
    def get_dummies(s, name):
        dummies = pd.get_dummies(s, prefix=name, prefix_sep=': ')
        dummies = dummies.drop(columns=dummies.sum().idxmax())
        return dummies
    
    X_categoric = [get_dummies(df[x], x) for x in categoric_independent_variables]
    X_base = pd.concat(X_regular + X_categoric, axis=1)
    if 'instruments' in kw_args:
        kw_args['instruments'] = df[kw_args['instruments']]
    if weights_variable is None:
        weights = None
    else:
        weights = df[weights_variable]
    if nesting_variable is not None:
        nesting_variable = df[nesting_variable]
    return  {y : linreg_internal(Y=df[y], 
                                 X=X_base, 
                                 weights=weights, 
                                 x_name=independent_variables+categoric_independent_variables, 
                                 y_name=y, 
                                 normalizeX=normalizeX, 
                                 normalizeY=normalizeY,
                                 nesting_variable=nesting_variable,
                                 df_with_const=df_with_const,
                                 **kw_args) 
                    for i, y in enumerate(dependent_variables)}

class EmptyResults:
    def __init__(self):
        self.rsquared = None
    
    def summary(self):
        return None

    def predict(self, x):
        return pd.Series()

def linreg_internal(Y, 
                    X, 
                    instruments=None, 
                    weights=None, 
                    iqr=1, 
                    normalizeX=True, 
                    normalizeY=True, 
                    df_with_const=False, 
                    nesting_variable=None,
                    **kw_args):
    # if 'x_name' in kw_args and 'y_name' in kw_args:
    #     logger.info("Running linear regression from %s to %s" % (kw_args['x_name'], kw_args['y_name']))
    assert instruments is None or nesting_variable is None
    assert weights is None or nesting_variable is None
    Y = reject_outliers(Y, iqr=iqr)
    X = X.apply(lambda col: reject_outliers(col, iqr=iqr))
    X = X.dropna()
    if len(X) == 0 or len(Y) == 0:
        return {'X' : X, 
                'Y' : Y, 
                'results' : EmptyResults(), 
                'results_df' : pd.DataFrame(columns=['coef', 'error', '[0.025'], 
                                            index=list(X.columns)+['const'])}
    
    
    if normalizeY:
        Y = pd.Series(preprocessing.MinMaxScaler().fit_transform(Y.values.reshape(-1, 1)).T[0], 
                        index=Y.index, 
                    name=Y.name)
    if normalizeX:
        if X.shape[1] == 1:
            X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X.values.reshape(-1, 1)).T[0], 
                             index=X.index, 
                             columns=X.columns)
        else:
            X = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(X), 
                                    index=X.index, 
                                    columns=X.columns)

        if instruments is not None: 
            if instruments.shape[1] == 1:
                instruments = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(instruments.values.reshape(-1, 1)).T[0], 
                                    index=instruments.index, 
                                    columns=instruments.columns)
            else:
                instruments = pd.DataFrame(preprocessing.MinMaxScaler().fit_transform(instruments), 
                                    index=instruments.index, 
                                    columns=instruments.columns)
    
    X = sm.add_constant(X)
    if instruments is not None:
        t1 = pd.concat([X, Y, instruments], axis=1)
    else:
        t1 = pd.concat([X, Y], axis=1)
    
    t = t1.dropna()
    if len(t) == 0:
        return {'X' : X, 'Y' : Y, 'results' : EmptyResults(), 'results_df' : pd.DataFrame()}
    X = t[X.columns]
    Y = t[Y.name]   
    if instruments is not None:
        instruments = t[instruments.columns]
    if weights is not None:
        weights = weights.loc[weights.index.intersection(X.index)]
        weights = weights[weights.notnull()]
        X = X.loc[weights.index]
        Y = Y.loc[weights.index]
    if nesting_variable is not None:
        nesting_variable = nesting_variable.loc[t.index]
    
    is_mixedLM=False

        
    if instruments is not None:
        model = IV2SLS(Y, X, instruments)
    elif nesting_variable is not None:
        model = sm.MixedLM(Y, X, groups=nesting_variable)
        is_mixedLM = True
    elif weights is None:
        model = sm.OLS(Y, X)
    else:
        model = sm.WLS(Y, X, weights)

    
    results = model.fit()
    results_df = linreg_results2df(results, 
                                    with_const=df_with_const, 
                                    is_mixedLM=is_mixedLM)
    

    return {'X' : X, 'Y' : Y, 'results' : results, 'results_df' : results_df}

def linreg_results2df(results, table_data=None, with_const=False, is_mixedLM=False, coef_col = 'coef'):
    '''
    Takes in results of OLS model and returns a plot of 
    the coefficients with 95% confidence intervals.
    
    Removes intercept, so if uncentered will return error.
    '''
    # Create dataframe of results summary 
    
    if is_mixedLM:
        results_df = results.summary().tables[1].rename(columns={'Coef.' : coef_col}).replace({"" : 0}) 
    else:
        if table_data is None:
            table_data = results.summary().tables[1].data
        results_df = pd.DataFrame(table_data)
        
        # Add column names
        results_df.columns = results_df.iloc[0]

        # Drop the extra row with column labels
        if 0 in results_df.index:
            results_df=results_df.drop(0)
         # Set index to variable names 
        results_df = results_df.set_index(results_df.columns[0])


    # Change datatype from object to float
    results_df = results_df.astype(float)

    # Get errors; (coef - lower bound of conf interval)
    errors = results_df[coef_col] - results_df['[0.025']
    
    # Append errors column to dataframe
    results_df['errors'] = errors

    # Drop the constant for plotting
    if 'const' in results_df.index and not with_const:
        results_df = results_df.drop(['const'])

    #results_df *= 100

    return results_df

IQR = 'iqr'
ZSCORE = 'zscore'
def reject_outliers(s, iqr=0.99,  method=IQR, zscore=3):
    s = s.dropna()
    if method == IQR:
        if iqr == 1:
            return s
        if len(s.value_counts()) == 2: 
            return s
        pcnt = (1 - iqr) / 2
        qlow, median, qhigh = s.dropna().quantile([pcnt, 0.50, 1-pcnt])
        iqr = qhigh - qlow
        loc = (s - median).abs() <= iqr
        return s[loc]
    if method == ZSCORE:
        locs = pd.Series(True, index=s.index)
        locs.loc[s.dropna().index] = (np.abs(stats.zscore(s.dropna())) > zscore)
        s[locs] = np.nan
        return s
    else:
        raise ValueError("REceived illegal outlier rejection method %s" % method)