import os
import sys
import numpy as np
import pandas as pd
import re
import json
from tqdm import tqdm
from collections import OrderedDict
import re

def load_json(fp):
    if not os.path.exists(fp):
        return dict()

    with open(fp, 'r', encoding='utf8') as f:
        return json.load(f)

def get_main_dir():
    if hasattr(sys, 'frozen'):
        return os.path.join(os.path.dirname(sys.executable))
    return os.path.join(os.path.dirname(__file__), '..')

def get_abs_path(*name):
    return os.path.abspath(os.path.join(get_main_dir(), *name))



