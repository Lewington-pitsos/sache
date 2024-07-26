import json
import os
from uuid import uuid4

LOG_DIR = 'log'

class ProcessLogger():
    def __init__(self, cache_dir):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

        self.cache_dir = cache_dir
        self.logger_id = str(uuid4())

        print('Logging to', self._log_filename())

    def _log_filename(self):
        return os.path.join(LOG_DIR,  self.cache_dir + '_' + self.logger_id + '.jsonl')

    def log(self, data):
        with open(self._log_filename(), 'a') as f:
            f.write(json.dumps(data) + '\n')