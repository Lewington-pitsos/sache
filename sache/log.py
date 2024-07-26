import time
import json
import os
from uuid import uuid4

LOG_DIR = 'log'

class ProcessLogger():
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.logger_id = str(uuid4())

        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR, exist_ok=True)

        print('Logging to', self._log_filename())

    def _log_filename(self):
        return os.path.join(LOG_DIR,  self.cache_dir + '_' + self.logger_id + '.jsonl')

    def log(self, data):
        if 'timestamp' not in data:
            data['timestamp'] = time.time()
        if 'hr_timestamp' not in data:
            data['hr_timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(data['timestamp']))

        with open(self._log_filename(), 'a') as f:
            f.write(json.dumps(data) + '\n')