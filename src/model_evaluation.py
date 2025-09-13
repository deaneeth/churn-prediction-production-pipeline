import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Union
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import logging
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)