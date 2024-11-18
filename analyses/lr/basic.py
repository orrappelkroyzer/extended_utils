
import sys
from pathlib import Path
local_python_path = str(Path(__file__).parents[3])
if local_python_path not in sys.path:
   sys.path.append(local_python_path)
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json") 

import pandas as pd
import plotly.express as px
from utils.plotly_utils import fix_and_write
from extended_utils.utils import float2str
from extended_utils.tools.linreg_tools import linreg
from extended_utils.analyses.lr.linreg_savers import write_coeffs, save_linreg
import numpy as np
import plotly.colors as pc


def get_dummies(s, name):
    dummies = pd.get_dummies(s, prefix=name, prefix_sep=': ')
    dummies = dummies.drop(columns=dummies.sum().idxmax())
    return dummies

def linreg_analysis(df, 
           independent_variables,
           dependent_variables,
           output_dir,
           categoric_independent_variables=[],
           intersection_variables=[],
           filename=None,
           normalizeX=True, 
           normalizeY=True,
           weights_variable=None,
           nesting_variable=None,
            **kw_args):
    
    #logger.info("running linreg")
    linreg_dict = linreg(df,
                        independent_variables,
                        dependent_variables,
                        categoric_independent_variables=categoric_independent_variables,
                        intersection_variables=intersection_variables,
                        normalizeX=normalizeX,
                        normalizeY=normalizeY,
                        weights_variable=weights_variable,
                        nesting_variable=nesting_variable,
                        **kw_args)

    #logger.info("linreg completed")
    if filename is not None:
       save_linreg(linreg_dict, output_dir, filename)
    return linreg_dict
    
def coeffs_analysis(df, 
                    output_dir,
                    dependent_variables,
                    linreg_results = None,
                    independent_variables=[],
                    categoric_independent_variables=[],
                    intersection_variables=[],
                    uninteresting_independent_variables=['const'],
                    normalizeX=True, 
                    normalizeY=True,
                    filename=None,
                    title="",
                    plotter=None,
                    ax_index=None,
                    nesting_variable=None,
                    weights_variable=None,
                    specified_iv = None,
                    sort_values=True,
                    width_factor=1,
                    height_factor=1,
                    no_colors=False,
                    facet_col_wrap=0,
                    **kw_args):
    
    assert plotter is None or (ax_index is not None and (len(dependent_variables) == 1) or linreg_results is not None)
    
    if linreg_results is None:
        linreg_results = linreg_analysis(df=df, 
                            output_dir=output_dir,
                            independent_variables=independent_variables,
                            categoric_independent_variables=categoric_independent_variables,
                            intersection_variables=intersection_variables,
                            dependent_variables=dependent_variables,
                            filename=filename,
                            normalizeX=normalizeX,
                            normalizeY=normalizeY,
                            nesting_variable=nesting_variable,
                            weights_variable=weights_variable,
                            **kw_args)
    write_coeffs(linreg_dict=linreg_results, output_dir=output_dir, filename=filename)
    uivs = []
    for uiv in uninteresting_independent_variables:
        if uiv in independent_variables:
            uivs += [uiv]
        elif uiv in categoric_independent_variables:
            uivs +=  list(get_dummies(df[uiv], uiv).columns)
        elif uiv == 'const':
            uivs += [uiv]
        else:
            raise AssertionError(f"Received uiv {uiv} which isn't an iv")
    coeffs_df = {}
    fig_dfs = []
    fixed_text = {}

    for name in dependent_variables:
        result= linreg_results[name]
        kw_args = dict(kw_args) 
        coeffs_df[name] = result['results_df'][['coef', 'errors']]
        last_fig_df = coeffs_df[name].loc[coeffs_df[name].index.drop(coeffs_df[name].index.intersection(set(uivs)))]
        fig_dfs += []
        last_fig_df['iv'] = last_fig_df.index
        last_fig_df['dv'] = name
        fixed_text[name] = kw_args.get('fixed_text', {})    
        if 'text' in fixed_text[name]:
            fixed_text[name] += "\nR^2 = %s" % round(result['results'].rsquared, 2)
        else:
            fixed_text[name] = "R^2 = %s" % round(result['results'].rsquared, 2)
        fixed_text[name] += ", {} datapoints".format(float2str(len(result['X'])))
        
        # if 'P>|t|' in result[0].columns:
        #     kw_args['faded_values']=list(result[0][result[0]['P>|t|'] > 0.05].index)
        if last_fig_df['errors'].dropna().empty:
            raise AssertionError("LR failed")
       

        fig_dfs += [last_fig_df]
    
    fig_df = pd.concat(fig_dfs)
    fig_df['significant'] = np.sign(fig_df['coef'] - fig_df['errors']) == np.sign(fig_df['coef'] + fig_df['errors'])
    category_orders={'dv' : dependent_variables}
    if sort_values:
        fig_df['abs_coef'] = np.abs(fig_df['coef'])
        fig_df = fig_df.sort_values(by='abs_coef', ascending=False)
        fig_df['iv'] = pd.Categorical(fig_df['iv'], categories=fig_df['iv'].unique(), ordered=True)
        category_orders['iv'] = fig_df['iv'].cat.categories

    # if specified_iv:
    #     fig_df_index = fig_df.index.tolist()
    #     fig_df_index.remove(specified_iv)
    #     fig_df_index = [specified_iv] + fig_df_index
    #     fig_df = fig_df.loc[fig_df_index]
    fig_df.index = range(len(fig_df))
    if no_colors:
        color_discrete_sequence = {True: "blue", False: "orange"}
        color = 'significant'
    else:
        N = fig_df['iv'].nunique()
        color_discrete_sequence = pc.sample_colorscale("rainbow", [i / (N - 1) for i in range(N)])
        color = 'iv'
    fig = px.scatter(fig_df, 
                     x='coef', 
                     y='iv', 
                     color=color, 
                     error_x='errors',
                     facet_col='dv',
                     facet_col_wrap=facet_col_wrap, 
                     category_orders=category_orders,
                     color_discrete_sequence  = color_discrete_sequence,
                     title=title)
    fig.update_layout(xaxis_title="Coefficient")
    fig.add_vline(x=0, fillcolor='dark grey')
    fig.update_annotations(font_size=24)
    fixed_text_params = dict(xref="paper", yref="paper",
                            x=0, y=0, showarrow=False)
    fixed_text_params.update(kw_args.get('fixed_text', {}).items())
    fixed_text_params['text'] = "<br>".join([": ".join(t) for t in fixed_text.items()])
    fixed_text_params['font'] = dict(size=24)
    
    fig.add_annotation(**fixed_text_params)
    fix_and_write(fig=fig, 
                 traces=dict(marker=dict(size=10)),
                 layout_params=dict(showlegend=False),
                 filename=filename,
                 width_factor=width_factor,
                 height_factor=height_factor,
                 output_dir=output_dir)
        
        
