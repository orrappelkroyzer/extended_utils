import os, sys
import json
import pandas as pd
import numpy as np
from datetime import datetime

local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-4])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)




import logging
from utils.utils  import LOG_LEVEL
logging.basicConfig(format='%(asctime)s|%(levelname)s|%(name)s (%(lineno)d): %(message)s', datefmt="%d/%m/%y %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)
from time import sleep
model_count = 1
from pathlib import Path

with open(os.path.join(local_python_path, "config.json")) as f:
    config = json.load(f)

def save_linreg(linreg_dict, output_dir, filename):

    global model_count
    for i in range(3):
        try:
            filename = Path(output_dir) / "internal" / filename.replace("/", "_")
            if not os.path.exists(os.path.split(filename)[0]):
                os.makedirs(os.path.split(filename)[0])
            for k, results in linreg_dict.items():
                # logger.info(f"Saving model {k} for {filename}")
                # with open(f"{filename}_{k}.txt", 'w') as f:
                #     f.write(str(results['results'].summary()))
                #     f.close()
                logger.info(f"Saving model {k} to {output_dir / f'{filename}_models.txt'}")
                with open(output_dir / "all_models.txt", 'a') as f:
                    f.write(f"Model number {model_count}, {filename.stem}\n".replace("_", " ").upper())
                    f.write(str(results['results'].summary()))
                    f.write("\n++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++\n")
                    model_count += 1
            return
        except:
            logger.info("Failed to save model, retrying")
            import traceback
            logger.error(traceback.format_exc())
            sleep(1)
            continue

def write_coeffs(linreg_dict, output_dir, filename):
    
    summary = pd.DataFrame()
    for i, (dv, result) in enumerate(linreg_dict.items()):
        def format_row(row):
            if (not np.isnan(row['coef']) and not np.isnan(row['errors'])):
                return "{} ({})".format(round(row['coef'], 2), round(row['errors'], 2))
            else:
                return 
        summary[dv] = result['results_df'].apply(format_row, axis=1)
        if 'P>|t|' in result['results_df'].columns:
            summary.loc[result['results_df']['P>|t|'] < 0.05, dv] = summary.loc[result['results_df']['P>|t|'] < 0.05, dv].apply(lambda x: "%s***" % x)
    filename = os.path.join(output_dir, "internal", "%s.csv" % filename.replace("/", " "))    
    if not os.path.exists(os.path.split(filename)[0]):
        os.makedirs(os.path.split(filename)[0])
    if filename is None:
        return summary
    summary.to_csv(filename)

def write_df(linreg_dict, filename):
    
    summary = pd.DataFrame()
    for i, (dv, result) in enumerate(linreg_dict.items()):
        summary["{}_coef".format(dv)] = result['results_df']['coef']
        summary["{}_errors".format(dv)] = result['results_df']['errors']
    filename = os.path.join(output_dir, "%s.csv" % filename.replace("/", " "))
    summary.to_csv(filename)

        