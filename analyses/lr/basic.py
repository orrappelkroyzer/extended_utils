
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
from utils.plotly_utils import fix_and_write, get_colors
from extended_utils.utils import float2str
from extended_utils.tools.linreg_tools import linreg
from extended_utils.analyses.lr.linreg_savers import write_coeffs, save_linreg



def linreg_analysis(df, 
           independent_variables,
           dependent_variables,
           output_dir,
           categoric_independent_variables=[],
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
                        normalizeX=normalizeX,
                        normalizeY=normalizeY,
                        weights_variable=weights_variable,
                        nesting_variable=nesting_variable,
                        **kw_args)

    #logger.info("linreg completed")
    if filename is not None:
       save_linreg(linreg_dict, output_dir, filename)
    return linreg_dict
    

# def coeff_surface_analysis(df, 
    #                        coeffs, 
    #                        X,
    #                        Y, 
    #                        stratifier,
    #                        n_strata,
    #                        title="",
    #                        filename=None,
    #                        cmap='bwr',
    #                        x_range=None,
    #                        y_range=None,
    #                        normalizeX=False,
    #                        **kw_args):
    # cols = list(coeffs.index.drop('const'))
    # if normalizeX:
    #     assert False
    #     if len(cols) == 1:
    #         df = pd.DataFrame(prepython_MinMaxScaler().fit_transform(df[cols].values.reshape(-1, 1)).T[0], 
    #                          index=df.index, 
    #                          columns=cols)
    #     else:
    #         df = pd.DataFrame(prepython_MinMaxScaler().fit_transform(df[cols]), 
    #                                 index=df.index, 
    #                                 columns=cols)
    # plotter = SurfacePlotter(num_plots = n_strata)
    # const = coeffs['const']

    # for k, v in coeffs.iteritems():
    #     if k in [X, Y, stratifier, 'const']:
    #         continue
    #     const += v*df[k].mean()
    # if normalizeX:
    #     assert False
    #     stratifer_consts = [df.loc[df[stratifier].between(i/n_strata, (i+1)/n_strata), 
    #                           stratifier].mean()*coeffs[stratifier]
    #                     for i in range(n_strata)]
    #     x_range = [0, 1]
    #     y_range = [0, 1]
        
    # else:
    #     stratifer_consts = [df.loc[df[stratifier].between(df[stratifier].quantile(i/n_strata),
    #                                                  df[stratifier].quantile((i+1)/n_strata)), 
    #                           stratifier].mean()*coeffs[stratifier]
    #                     for i in range(n_strata)]
    #     logger.info(stratifer_consts)
    
    # if x_range is None:
    #     x_range = [df[X].min(), df[X].max()]
    # if y_range is None:
    #     y_range = [df[Y].min(), df[Y].max()]
    # t_df = pd.DataFrame({'x' : x_range, 'y' : y_range}, index=['min', 'max']).T
    # for i in range(n_strata):
    #     plot_params = dict(z_func=lambda x, y: coeffs[X]*x+coeffs[Y]*y+const+stratifer_consts[i],
    #                         cmap=cmap,
    #                         subtitle="{} quantile {}/{} ({}-{})".format(stratifier, i+1, n_strata, df[stratifier].quantile(i/n_strata), df[stratifier].quantile((i+1)/n_strata)),
    #                         xlabel=X,
    #                         ylabel=Y,
    #                         filename=filename)   
    #     if 'norm' in kw_args:
    #         plot_params['norm']=kw_args['norm']
    #     plotter.plot(df=t_df, ax_index=i, **plot_params)

def coeffs_analysis(df, 
                    output_dir,
                    dependent_variables,
                    linreg_results = None,
                    independent_variables=[],
                    categoric_independent_variables=[],
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
                    **kw_args):
    
    assert plotter is None or (ax_index is not None and (len(dependent_variables) == 1) or linreg_results is not None)
    
    if linreg_results is None:
        linreg_results = linreg_analysis(df=df, 
                            output_dir=output_dir,
                            independent_variables=independent_variables,
                            categoric_independent_variables=categoric_independent_variables,
                            dependent_variables=dependent_variables,
                            filename=filename,
                            normalizeX=normalizeX,
                            normalizeY=normalizeY,
                            nesting_variable=nesting_variable,
                            weights_variable=weights_variable,
                            **kw_args)
    
    write_coeffs(linreg_dict=linreg_results, output_dir=output_dir, filename=filename)
    

    coeffs_df = {}
    fig_dfs = []
    fixed_text = {}
    for name in dependent_variables:
        result= linreg_results[name]
        kw_args = dict(kw_args) 
        coeffs_df[name] = result['results_df'][['coef', 'errors']]
        fig_dfs += [coeffs_df[name].loc[coeffs_df[name].index.drop(coeffs_df[name].index.intersection(uninteresting_independent_variables))]]
        fig_dfs[-1]['iv'] = fig_dfs[-1].index
        fig_dfs[-1]['dv'] = name
        fixed_text[name] = kw_args.get('fixed_text', {})    
        if 'text' in fixed_text[name]:
            fixed_text[name] += "\nR^2 = %s" % round(result['results'].rsquared, 2)
        else:
            fixed_text[name] = "R^2 = %s" % round(result['results'].rsquared, 2)
        fixed_text[name] += ", {} datapoints".format(float2str(len(result['X'])))
        
        # if 'P>|t|' in result[0].columns:
        #     kw_args['faded_values']=list(result[0][result[0]['P>|t|'] > 0.05].index)
        if fig_dfs[-1]['errors'].dropna().empty:
            raise AssertionError("LR failed")
    
    fig_df = pd.concat(fig_dfs)
    if sort_values:
        fig_df = fig_df.sort_values('coef', ascending=False)
    if specified_iv:
        fig_df_index = fig_df.index.tolist()
        fig_df_index.remove(specified_iv)
        fig_df_index = [specified_iv] + fig_df_index
        fig_df = fig_df.loc[fig_df_index]
    fig_df.index = range(len(fig_df))
    
    fig = px.scatter(fig_df, 
                     x='coef', 
                     y='iv', 
                     color='iv', 
                     error_x='errors',
                     facet_col='dv',
                     category_orders={'dv' : dependent_variables},
                     color_discrete_sequence  = get_colors(fig_df['iv'].nunique()),
                     title=title)
    fig.update_layout(xaxis_title="Coefficient")
    fig.add_vline(x=0, fillcolor='dark grey')
    fig.update_annotations(font_size=24)
    fixed_text_params = dict(xref="paper", yref="paper",
                            x=0, y=0, showarrow=False)
    fixed_text_params.update(kw_args.get('fixed_text', {}).items())
    fixed_text_params['text'] = "<br>".join([": ".join(t) for t in fixed_text.items()])
    fixed_text_params['font'] = dict(size=20)
    
    fig.add_annotation(**fixed_text_params)
    fix_and_write(fig=fig, 
                 traces=dict(marker=dict(size=10)),
                 layout_params=dict(showlegend=False),
                 filename=filename,
                 width_factor=width_factor,
                 height_factor=height_factor,
                 output_dir=output_dir)
        
        
