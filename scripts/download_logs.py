import os 
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sache.log import download_logs
from sache.constants import BUCKET_NAME


download_logs(BUCKET_NAME)