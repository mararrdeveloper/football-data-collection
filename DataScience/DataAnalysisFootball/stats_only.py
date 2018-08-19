import pandas as pd
import numpy as np
import helpers

from helpers import test_clfs, load_data, print_results, get_baseline

X,target=load_data(columns_to_drop=['player0_B365','player0_Aces','player0_PS',
'player0_EX','player0_LB','player1_B365','player1_Aces','player1_PS','player1_EX','player1_LB'])

res=test_clfs(clfs=helpers.clfs,X=X,target=target,cv=10)

get_baseline()

print_results(res)
