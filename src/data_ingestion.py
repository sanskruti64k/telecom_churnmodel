# import Data Manipulation Libraries

import numpy as np
import pandas as pd

def data_ingestion():
    df = pd.read_csv(r'https://raw.githubusercontent.com/sanskruti64k/telecom_churnmodel/refs/heads/main/data/churn.csv')

    return df