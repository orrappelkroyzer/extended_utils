import spacy
nlp = spacy.load("en_core_web_lg")
import pandas as pd
import os, sys, json

local_python_path = os.path.sep.join(__file__.split(os.path.sep)[:-4])
if local_python_path not in sys.path:
    sys.path.append(local_python_path)


import logging
from python_utils.utils  import LOG_LEVEL
logging.basicConfig(format='%(asctime)s|%(levelname)s|%(name)s (%(lineno)d): %(message)s', datefmt="%d/%m/%y %H:%M:%S")
logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)


with open(os.path.join(local_python_path, "config.json")) as f:
    config = json.load(f)
my_dpi = config['my_dpi']


default_ignored_words = ['-PRON-', "a", "the", "an", "the", "to", "in", "for", "of", "or", "by", "with", "is", "on", "that", "be"]
ignored_pos = ['ADP','AUX', 'CCONJ', 'DET', 'PART', 'PRON', 'PUNCT', 'SCONJ', 'SYM']

from python_utils.plotters.wordcloud_plotters import WordCloudPlotter

def wordcloud_analisys(df, output_dir, filename, ignored_words=[]):
    logger.info("wordcloud analysis")
    ignored_words += default_ignored_words
    lemmas = []
    for doc in df:
        lemmas += [x.lemma_.lower() for x in doc if x.is_alpha and x.pos_ not in ignored_pos]

    hist = pd.Series(lemmas).value_counts().drop(ignored_words, errors='ignore')
    WordCloudPlotter(output_dir=output_dir).plot(df=hist,
                                                colormap='RdYlGn',
                                                 filename=filename)