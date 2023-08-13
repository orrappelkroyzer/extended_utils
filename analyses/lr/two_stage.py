

import pandas as pd
import numpy as np
import statsmodels.api as sm


import sys
from pathlib import Path
local_python_path = str(Path(__file__).parents[3])
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json") 


from extended_utils.tools.linreg_tools import  linreg_results2df
from extended_utils.analyses.lr.basic import linreg_analysis, coeffs_analysis




def linreg_2_stage_analysis(df, 
            independent_variables,
           intermediate_variable,
           dependent_variable,
           output_dir,
           filename,
           uninteresting_independent_variables=[],
           title="",
           # surface_params=False,
           stage1_mask=None,
           stage2_mask=None,
           no_plot=False,
           **kw_args):
    coeffs, uiv, r2, xInd = linreg_2_stage_internal(df=df, 
            independent_variables=independent_variables,
           intermediate_variable=intermediate_variable,
           dependent_variable=dependent_variable,
           uninteresting_independent_variables=uninteresting_independent_variables,
           stage1_mask=stage1_mask,
           stage2_mask=stage2_mask,
           output_dir=output_dir,
           filename=filename,
           **kw_args)
    if no_plot:
        table_data = [['', 'coef','[0.025']] + [[coeffs.index[i]] + list(coeffs.iloc[i].values) for i in range(len(coeffs))]
        class R2_wrapper:
            def __init__(self, r2): self.rsquared = r2
        return {'results_df' : linreg_results2df(None, table_data=table_data), 'results' : R2_wrapper(r2), 'X' : xInd}
   # if surface_params:
    #     coeff_surface_analysis(df=df, coeffs=coeffs['coef'], title=title, **surface_params)
    else:
        if len(title) > 0:
            title = "%s (through %s)" % (title, intermediate_variable)
        table_data = [['', 'coef','[0.025']] + [[coeffs.index[i]] + list(coeffs.iloc[i].values) for i in range(len(coeffs))]
        coeffs_analysis(df=df,
                        output_dir=output_dir,
                        linreg_results = {dependent_variable : (linreg_results2df(None, table_data), r2, xInd)},
                        dims = [1280, 623],
                        categoric_independent_variables=[],
                        uninteresting_independent_variables=uiv,
                        sort_values=independent_variables,
                        filename=filename,
                        title=title)

def linreg_2_stage_internal(df, 
            independent_variables,
           intermediate_variable,
           dependent_variable,
           filename,
           output_dir,
           uninteresting_independent_variables=[],
           stage1_mask=None,
           stage2_mask=None,
           normalizeX=True,
           **kw_args):
    if stage1_mask is None:
        stage1_mask=pd.Series(True, index=df.index)
    if stage2_mask is None:
        stage2_mask=pd.Series(True, index=df.index)
    stage_1_predicted_df, stage2_iv, stage_1_coeffs = \
        two_stage_run_first_stage(df=df, 
                                  independent_variables=independent_variables, 
                                  intermediate_variable=intermediate_variable, 
                                  normalizeX=normalizeX,
                                  filename=filename,
                                  output_dir=output_dir,
                                  **kw_args)
    predicted, stage_2_coeffs = \
        two_stage_run_second_stage(df=df, 
                                   stage2_iv=stage2_iv, 
                                   dependent_variable=dependent_variable,
                                   stage_1_predicted_df=stage_1_predicted_df, 
                                   filename=filename,
                                   output_dir=output_dir,
                                   **kw_args)
    coeffs, uiv = two_stage_create_coeff_table(stage_1_coeffs, 
                                 stage_2_coeffs, 
                                 intermediate_variable, 
                                 uninteresting_independent_variables, 
                                 **kw_args)
    actual = df.loc[predicted.index, dependent_variable[0]]
    r2 = max(1-((actual-predicted)**2).sum()/((actual-actual.mean())**2).sum(), 0)
    return coeffs, uiv, r2, actual.index

def two_stage_run_first_stage(df, independent_variables, intermediate_variable, normalizeX, output_dir, filename, **kw_args):
    linreg_dict = linreg_analysis(df=df, 
            independent_variables=independent_variables,
           dependent_variables=[intermediate_variable],
            normalizeX = normalizeX, 
            normalizeY=False,
           return_full_results = True,
           df_with_const=True,
           output_dir=output_dir,
           filename=f"{filename}_stage1",
           **kw_args)[intermediate_variable]

    stage_1_predicted_df = pd.DataFrame()
    stage_1_predicted_df[intermediate_variable] = linreg_dict['results'].predict(sm.add_constant(linreg_dict['X']))
    return stage_1_predicted_df, [intermediate_variable], linreg_dict['results_df']

def two_stage_run_second_stage(df, stage2_iv, dependent_variable, stage_1_predicted_df, output_dir, filename, **kw_args):
    linreg_dict = linreg_analysis(df=df, 
            independent_variables=stage2_iv,
            dependent_variables=dependent_variable,
            return_full_results = True,
            normalizeX=False,
            normalizeY=False,
            df_with_const=True,
            filename=f"{filename}_stage2",
            output_dir=output_dir,
            **kw_args)[dependent_variable[0]]
    
   
    stage_1_predicted_df = stage_1_predicted_df.loc[stage_1_predicted_df.index.intersection(linreg_dict['X'].index)]
    predicted = linreg_dict['results'].predict(sm.add_constant(stage_1_predicted_df))  
    return predicted, linreg_dict['results_df']

def two_stage_create_coeff_table(stage_1_coeffs, stage_2_coeffs, intermediate_variable, uninteresting_independent_variables, **kw_args):
    coeffs = pd.DataFrame({'coef' : stage_1_coeffs['coef']*stage_2_coeffs.loc[intermediate_variable, 'coef'],
                             '[0.025' : stage_1_coeffs.apply(lambda row: \
                                                             row[['[0.025', '0.975]']].apply(lambda x: 
                                                                x*stage_2_coeffs.loc[intermediate_variable, ['[0.025', '0.975]']].min()).min(),
                                                                    axis=1)}) 
    coeffs.loc['const'] += stage_2_coeffs.loc['const']
    
    # coeffs.loc['const'] += stage_2_coeffs.loc['const', ['coef', '[0.025']]
    uiv = list(uninteresting_independent_variables)
    return coeffs, uiv
  