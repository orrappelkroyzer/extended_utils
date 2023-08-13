

import os, sys
import json
import pandas as pd

local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-4])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)


from utils.utils import LOG_LEVEL
from extended_utils.analyses.lr.basic import linreg_analysis
from extended_utils.analyses.lr.two_stage import linreg_2_stage_analysis
from extended_utils.analyses.lr.linreg_savers import write_coeffs
from utils.plotly_utils import fix_and_write, get_colors
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from pathlib import Path
from utils.utils import load_config, get_logger
logger = get_logger(__name__)
config = load_config(Path(local_python_path) / "config.json") 
my_dpi = config['my_dpi']
import numpy as np

def darken_color(rgb, factor):
    rgb = [int(y) for y in rgb[4:-1].split(",")]
    r, g, b = rgb
    return f"rgb({', '.join(tuple(str(int(max(0, min(255, c * factor)))) for c in (r, g, b)))})"


def coeff_compare_control_analysis(df, 
                    independent_variables,
                    dependent_variables,
                    control_variable,
                    output_dir,
                    control_variable_index=2,
                    uninteresting_independent_variables=[],
                    replacement_variables={},
                    title="",
                    filename=None,
                    fixed_text = {},
                    width_factor=1,
                    height_factor=1,
                    horizontal_spacing = None,
                    vertical_spacing = None,
                    intermediate_variables=None,
                    nesting_variable=None,
                    weights_variable=None,
                    **kw_args):
    
    
    rename_dicts = {i : {v[i] : k for k, v in replacement_variables.items()} for i in ['non_control', 'control']}
    ivs = {'non_control' : independent_variables, 
           'control' : independent_variables[:control_variable_index] + \
                                                              [control_variable] +\
                                                              independent_variables[control_variable_index:]}
    if intermediate_variables is None:
        linreg_func = lambda k, kw_args: linreg_analysis(df=df.rename(rename_dicts[k], axis=1), 
                                          independent_variables=ivs[k],
                                          dependent_variables=dependent_variables,
                                          categoric_independent_variables=[],
                                          filename=f"{filename}_{k}",
                                          nesting_variable=nesting_variable,
                                          weights_variable=weights_variable,
                                          output_dir=output_dir,
                                          **kw_args)
    else:
        linreg_func = lambda k, kw_args: {int_v : linreg_2_stage_analysis(df=df.rename(rename_dicts[k], axis=1), 
                                                                independent_variables=ivs[k],
                                                                intermediate_variable=int_v,
                                                                dependent_variable=dependent_variables,
                                                                filename=f"{filename}_{k}_{int_v}",
                                                                no_plot=True,
                                                                output_dir=output_dir,
                                                                nesting_variable=nesting_variable,
                                                                weights_variable=weights_variable,
                                                                **kw_args) for int_v in intermediate_variables}

        

    
    linreg_results = {k : linreg_func(k, kw_args) 
                      for k in ['control', 'non_control']}
    
    
    for k in linreg_results['non_control'].keys():
        linreg_results['non_control'][k]['results_df'] = \
            pd.concat([linreg_results['non_control'][k]['results_df'].iloc[:control_variable_index+1], 
                        pd.DataFrame(0, index=[control_variable], columns=linreg_results['non_control'][k]['results_df'].columns), 
                       linreg_results['non_control'][k]['results_df'].iloc[control_variable_index+1:]])
                       
    
    for k, v in linreg_results.items():
        write_coeffs(linreg_dict=v, output_dir=output_dir, filename="%s_%s" % (filename, k))

    colors = get_colors(len(independent_variables))
    colors = {'non_control' :  colors,
             'control' : [darken_color(c, 0.75) for c in colors]}
 
    plots = dependent_variables
    if intermediate_variables is not None:
        plots = intermediate_variables
    if len(plots) > 1:
        subplots_args = dict(rows=int((len(plots)-1)/2+1), 
                             cols=2, 
                             shared_yaxes=True)
    else:   
        subplots_args = dict(rows=len(plots), cols=1)
    if len(plots) > 1:
        subplots_args['subplot_titles'] = plots
    subplots_args['vertical_spacing'] = vertical_spacing
    subplots_args['horizontal_spacing'] = horizontal_spacing
    fig = make_subplots(**subplots_args)
    fig.update_annotations(font_size=24)
    fixed_text = {}
    
    for i, dv in enumerate(linreg_results['non_control']):    
        row=int(i/2)+1
        col=i%2+1
        for k in ['non_control', 'control']:         
            result = linreg_results[k][dv]
            kw_args = dict(kw_args) 
            s = result['results_df']['coef']
            s = s[s.index.drop(uninteresting_independent_variables)]
            std=result['results_df']['errors'][s.index]       
            if k == 'non_control':
                fixed_text[dv] = kw_args.get('fixed_text', {})
                if 'text' in fixed_text[dv]:
                    fixed_text[dv] += f"\nR^2 = {round(result['results'].rsquared)}"
                else:
                    fixed_text[dv] = f"R^2 = {round(result['results'].rsquared)}"
                
                fixed_text[dv] += f", {len(result['X'])} datapoints"
            for t in range(len(s)-1, -1, -1):
                fig.add_trace(go.Scatter(
                    x=[s.values[t]],
                    y=[s.index[t]],
                    error_x=dict(
                        type='data',
                        array=[std.values[t]],
                        color=colors[k][t], # use a single color for each trace
                        thickness=1,
                        width=3
                    ),
                    mode='markers',
                    marker=dict(color=colors[k][t]),
                    showlegend=False,
                ), \
                row=row, col=col)

           
        x=linreg_results['control'][dv]['results_df']['coef'].iloc[0]
        y=linreg_results['control'][dv]['results_df'].index[0]
        ax=linreg_results['non_control'][dv]['results_df']['coef'].iloc[0]
        ay=linreg_results['non_control'][dv]['results_df'].index[0]
        sign = np.sign(x-ax)
        epsilon = abs(x-ax)/10
        fig.add_annotation(x=x-sign*(linreg_results['control'][dv]['results_df']['errors'].iloc[0]+epsilon), 
                            xref=f'x{i+1}',
                            y=y, 
                            yref=f'y{i+1}',
                            ax=ax+sign*(linreg_results['non_control'][dv]['results_df']['errors'].iloc[0]+epsilon),  
                            axref=f'x{i+1}',
                            ay=ay, 
                            ayref=f'y{i+1}',
                            arrowhead=1,  # arrowhead style
                            arrowsize=1,  # arrowhead size
                            arrowwidth=3,  # arrow width
        )
        row_ref = "" if row == 1 else row
        col_ref = "" if col == 1 else col 
        annotation_y = -0.15
        if row==int((len(linreg_results['non_control'])-1)/2)+1:
            fig.update_xaxes(title_text="Coefficient", row=row, col=col)
            annotation_y = 0.01
        fig.add_annotation(text=f"R^2: {round(linreg_results['non_control'][dv]['results'].rsquared,2)}, {len(linreg_results['non_control'][dv]['X'])} datapoints",
                            xref=f"x{row_ref} domain", yref=f"y{col_ref} domain",
                                x=0.1, y=annotation_y, showarrow=False,
                                font=dict(size=20),
                                row=row, col=col)
        
    
    fix_and_write(fig=fig, 
                 traces=dict(marker=dict(size=10)),
                 layout_params=dict(showlegend=False,
                                    title_text=title),
                 filename=filename,
                 width_factor=width_factor,
                 height_factor=height_factor,
                 output_dir=output_dir)
