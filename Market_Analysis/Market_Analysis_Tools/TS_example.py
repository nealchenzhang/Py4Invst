import numpy as np
import pandas as pd
import Market_Analysis.Market_Analysis_Tools.TS_Analysis

pickle_file = open('my_list.pkl', 'rb')
my_list2 = pickle.load(pickle_file)
print(my_list2)
pickle_file.close()