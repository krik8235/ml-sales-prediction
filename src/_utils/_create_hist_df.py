import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


import os
import pandas as pd
from datetime import datetime

from src._utils import retrieve_file_path


time_stamp = datetime.now().timestamp()
current_dir =  os.getcwd()
folder_path = os.path.join(current_dir, 'data', 'processed')
file_path, _ = retrieve_file_path(folder_path=folder_path)

df = pd.read_csv(file_path)
df_85123A = df[df['stockcode'] == '85123A']
price_volume_list = []
for index, row in df_85123A.iterrows():
    price_volume_list.append({
        'price': row['unitprice'],
        'sales': row['unitprice'] * row['quantity'],
        'volume': row['quantity']
    })

print(price_volume_list)
