import pandas as pd
import sys
sys.path.append('E:/Work/BirdCLEF2017/')

xlsx_file = sys.path[-1] + 'Info/metadata.xlsx'

df = pd.read_excel(xlsx_file, index_col=None)
x = df.groupby(['Order', 'Family', 'Genus', 'ClassId']).groups.keys()
x = list(x)
