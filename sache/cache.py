class Cache():
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        self.cache = []

    def __iter__(self):
        idx = 0
        while idx < len(self.cache):
            batch = None
            while batch is None: # keep taking the next batch until one is free
                batch =  self.borrow(idx)
                idx += 1

            yield batch

    def append(self, activations):
        self.cache.append(activations)

    def borrow(self, idx):
        pass

    def give_back(self, idx, activations=None):
        pass

