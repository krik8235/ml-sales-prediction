from dotenv import load_dotenv # type: ignore
load_dotenv(override=True)


import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore")
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)


# adjust the path to import the main scripts
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
