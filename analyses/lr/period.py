
import os, sys
import json
import pandas as pd
import numpy as np

local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-4])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)



from extended_utils.analyses.lr.basic import linreg_analysis
from extended_utils.analyses.lr.two_stage import linreg_2_stage_analysis
from pathlib import Path
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json") 


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go

from utils.plotly_utils import fix_and_write, get_colors

def period_coeffs_analysis(df, 
                           independent_variables,
                           dependent_variables,
                           periods_df,
                           output_dir,
                           categoric_independent_variables=[],
                           uninteresting_independent_variables=[],
                           intermediate_variables=None,
                           filename=None,
                           special_coeffs = None,
                           sort_values=True,
                           nesting_variable=None,
                           weights_variable=None,
                           with_R2 = False,
                           specified_iv=None,
                           **kw_args
                           ):



    if intermediate_variables is None:
        def linreg_func(period, kw_args): 
            return  linreg_analysis(df=df[df.Date.between(period['start'], period['end'])], 
                                          independent_variables=independent_variables,
                                          dependent_variables=dependent_variables,
                                          categoric_independent_variables=categoric_independent_variables,
                                          output_dir=output_dir,
                                          filename=filename,
                                          return_full_results=True,
                                          weights_variable=weights_variable,
                                          nesting_variable=nesting_variable,
                                          **kw_args)
            
        plot_names = list(dependent_variables)
    else:
        def linreg_func(period, kw_args): 
            return  {int_v : linreg_2_stage_analysis(df=df[df.Date.between(period['start'], period['end'])], 
                                                                independent_variables=independent_variables,
                                                                intermediate_variable=int_v,
                                                                dependent_variable=dependent_variables,
                                                                filename=f"{filename}_{period['name']}",
                                                                no_plot=True,
                                                                output_dir=output_dir,  
                                                                **kw_args) for int_v in intermediate_variables}            
        plot_names = list(intermediate_variables)

   
    
    coeffs = {x : {} for x in plot_names}
    std = {x : {} for x in plot_names}
    summary = {x : pd.DataFrame() for x in plot_names}
    R2 = {x : {} for x in plot_names}
    linreg_results_dict = {}
    n_datapoints = {x : [] for x in plot_names}
    for j, (i, row) in enumerate(periods_df.iterrows()):
        logger.info(row['name'])
        if df[df.Date.between(row['start'], row['end'])][dependent_variables].notnull().sum().min() < 10:
            continue
        if j % 10 == 0:
            for x in plot_names:
                coeffs[x] = coeffs[x].copy()
                std[x] = std[x].copy()
                summary[x] = summary[x].copy()
        temp_kw_args = {k : v for k, v in kw_args.items() if k != 'filename'}
        linreg_results = linreg_func(row, temp_kw_args)
        
        

        for plot_name, result in linreg_results.items():
            if result is None or result['results_df'].empty or 'errors' not in result['results_df'].columns:
                continue
            index = [x for x in result['results_df'].index if x not in uninteresting_independent_variables]
            coeffs[plot_name][row['name']] = result['results_df']['coef'].loc[index]
            std[plot_name][row['name']] = result['results_df']['errors'].loc[index]
            R2[plot_name][row['name']] = result['results'].rsquared
            n_datapoints[plot_name] += [len(result['X'])]
            summary[plot_name][row['name']] = result['results_df'].apply(lambda row: f"{round(row['coef'])} ({round(row['errors'])})" if (not np.isnan(row['coef']) and not np.isnan(row['errors'])) else "None" , axis=1)
            if 'P>|t|' in result['results_df'].columns:
                t = pd.Series(result['results_df']['P>|t|'] > 0.05, index=summary[plot_name][row['name']].index).fillna(False)
                summary[plot_name].loc[t, row['name']] = \
                    summary[plot_name].loc[t, row['name']].apply(lambda x: "<%s>" % x)
        
        # R2[plot_name] = pd.Series(R2[plot_name])
        linreg_results_dict[row['name']] = linreg_results


    
    for dv, sum_df in summary.items():    
        fn = os.path.join(output_dir, "internal", "{}_{}.csv".format(filename.replace("/", " "), dv))    
        if not os.path.exists(os.path.split(fn)[0]):
            os.makedirs(os.path.split(fn)[0])
        sum_df.to_csv(fn)

    
    for plot_name in coeffs:
        coeffs[plot_name] = pd.DataFrame(coeffs[plot_name]).T
        coeffs[plot_name]['dv'] = plot_name
        coeffs[plot_name]['Date'] = coeffs[plot_name].index
        coeffs[plot_name].index = coeffs[plot_name]['Date'].apply(lambda x: (x, plot_name))
        std[plot_name] = pd.DataFrame(std[plot_name]).T
        std[plot_name]['dv'] = plot_name
        std[plot_name]['Date'] = std[plot_name].index
        std[plot_name].index = std[plot_name]['Date'].apply(lambda x: (x, plot_name))
    coeffs = pd.concat(coeffs.values())
    std = pd.concat(std.values())
    coeff_names = list(coeffs.columns.difference(['Date', 'dv']))
    if sort_values:
        coeff_names = coeffs[coeff_names].mean().sort_values(ascending=False).index.tolist()
    if special_coeffs is not None:
        coeff_names = list(set(special_coeffs).intersection(set(coeff_names)))
    if specified_iv:
        coeff_names.remove(specified_iv)
        coeff_names = [specified_iv] + coeff_names
    error_up = coeffs.copy(deep=True)
    error_up[coeff_names] += std[coeff_names]
    error_down = coeffs.copy(deep=True)
    error_down[coeff_names] -= std[coeff_names]
    if not with_R2:
        R2 = None
    plot_over_time(coeff_names, coeffs, error_up, error_down, n_datapoints, filename, output_dir, R2)
    return linreg_results_dict

def plot_over_time(coeff_names, coeffs, error_up, error_down, n_datapoints, filename, output_dir, R2=None, horizontal_spacing=None, vertical_spacing=None):
    dvs = coeffs['dv'].unique().tolist()
    colors, colors_faded = get_colors(len(coeff_names), with_faded=True)
    n_rows = len(dvs)
    subplots_args = dict(rows=len(dvs), cols=1)
    if len(dvs) > 1 or R2 is not None:
        subplots_args['subplot_titles'] = dvs
    
    subplots_args['horizontal_spacing'] = horizontal_spacing
    subplots_args['vertical_spacing'] = vertical_spacing
    
    fig = make_subplots(**subplots_args)
    fig.update_annotations(font_size=24)
    for j, dv in enumerate(dvs):
        for i, coeff_name in enumerate(coeff_names):
            x = coeffs[coeffs.dv == dv].Date.values.tolist()
            y = coeffs.loc[coeffs.dv == dv, coeff_name].values.tolist()
            y_upper = error_up.loc[error_up.dv == dv, coeff_name].values.tolist()
            y_lower = error_down.loc[error_down.dv == dv, coeff_name].values.tolist()
            
            if j > 0:
                j_ref = j+1
                showlegend=False
            else:
                j_ref = ''
                showlegend=True
            fig.add_trace(go.Scatter(
                name=coeff_name,
                x=x,
                y=y,
                line=dict(color=colors[i], width=2),
                mode='lines',
                showlegend=showlegend
            ), row=j+1, col=1)
            fig.add_trace(go.Scatter(
                x=x+x[::-1], # x, then x reversed
                y=y_upper+y_lower[::-1], # upper, then lower reversed
                fill='toself',
                fillcolor=colors_faded[i],
                line=dict(color=colors_faded[i]),
                marker=dict(size=0),
                hoverinfo="skip",
                showlegend=False
            ), row=j+1, col=1)
            
            n = np.array(n_datapoints[dv])
            fig.add_annotation(text=f"{n.min()}-{n.max()} datapoints, {round(n.mean())} on average",
                            xref=f"x{j_ref} domain", yref=f"y{j_ref} domain",
                            font=dict(size=20),
                                x=0.01, y=0.01, showarrow=False,
                                row=j+1, col=1)
        fig.add_hline(y=0, line=dict(width=2, color='black'))
        if j==len(dvs)-1:
            fig.update_xaxes(title_text="Date", row=j+1, col=1)
        fig.update_xaxes(title_text="Coefficient", row=j+1, col=1)
    #fig.update_layout(plot_bgcolor = "white")
    # fig.update_xaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black', 
    #                  griddash='dash',
    #                  tickmode='array', tickvals=t, ticktext=[x.strftime('%m-%d') for x in t],
    #                  range=[min(t), max(t)]) 
    # fig.update_yaxes(showline=True, linewidth=1, linecolor='black', gridcolor='black',
    #                 griddash='dash')
    
    
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.05,
            xanchor="center",
            x=0.5,
            font=dict(size=20)
        )
    )

    fix_and_write(fig=fig, 
                 filename=filename,
                 height_factor=1.5,
                 output_dir=output_dir)
    
    if R2 is not None:
        R2 = pd.DataFrame(R2)
        R2['Date'] = R2.index
        fig = make_subplots(rows=1, cols=1)
        for j, dv in enumerate(R2.columns):
            if dv == 'Date':
                continue
            fig.add_trace(go.Scatter(x=R2['Date'],
                                     y=R2[dv],
                                     mode='lines',
                                     name=dv,
                                     line=dict(color=colors[j], width=2)))
        fig.update_layout(xaxis_title="Date", yaxis_title="R^2",
                             legend=dict(
                                orientation="h",
                                yanchor="top",
                                y=-0.2,
                                xanchor="center",
                                x=0.5,
                                font=dict(size=20)
                            )
                        )
        fix_and_write(fig=fig, 
                 filename=f"{filename}_R2",
                 height_factor=0.33,
                 layout_params=dict(title_text="R^2"),
                 output_dir=output_dir)
