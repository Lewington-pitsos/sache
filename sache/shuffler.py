import random
from collections import defaultdict
from sache.cache import Cache

class Shuffler():
    def __init__(self, cache):
        self.cache = cache
        self.shuffle_record = defaultdict(list)

    def choose_two_indices(self):
        a_idx = random.randint(0, len(self.cache) - 1)
        b_idx = random.randint(0, len(self.cache) - 1)

        return a_idx, b_idx

    def _shuffle(self, a, b):
        pass

    def continuous_shuffle(self):
        while True: # while amount of shuffling is less than a certain amount
            self.shuffle()

    def shuffle(self):
        while True:
            a_idx, b_idx = self.choose_two_indices()

            a = self.cache.borrow(a_idx)
            b = self.cache.borrow(b_idx)

            if a is not None and b is not None:
                break

        a, b = self._shuffle(a, b)
        
        self.cache.give_back(a_idx, a)
        self.cache.give_back(b_idx, b)

        self.shuffle_record[a_idx].append(b_idx)
        self.shuffle_record[b_idx].append(a_idx)
    
def shuffle(cache_dir):
    cache = Cache(cache_dir)
    shuffler = Shuffler(cache)
    shuffler.continuous_shuffle()