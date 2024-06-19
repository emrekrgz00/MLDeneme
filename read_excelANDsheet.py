import numpy as np
import pandas as pd
import glob

veri = pd.ExcelFile("../data/GES.xls")
print(veri.sheet_names)

datum = pd.read_excel("../data/GES.xls", sheet_name="AydinMen")

print(datum)