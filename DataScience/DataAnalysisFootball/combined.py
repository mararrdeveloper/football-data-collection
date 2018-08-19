import pandas as pd
import numpy as np
import helpers
from helpers import test_clfs, load_data, get_baseline, print_results

X,target=load_data()

res=test_clfs(clfs=helpers.clfs,X=X,target=target,cv=10)

get_baseline()

print_results(res)