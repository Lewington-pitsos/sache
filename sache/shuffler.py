import random
from sache.cache import Cache

class Shuffler():
    def __init__(self, cache, n_shuffles):
        self.cache = cache
        self.n_shuffles = n_shuffles

    def choose_two_indices(self):
        a_idx = random.randint(0, len(self.cache) - 1)
        b_idx = random.randint(0, len(self.cache) - 1)

        return a_idx, b_idx

    def shuffle_in_place(self, activations, loaded):
        pass

    def shuffle(self, id):
        activations = self.cache.take(id)

        loaded_id, loaded = self.cache.take_random()
        self.shuffle_in_place(activations, loaded)

        self.cache.give_back(loaded_id, loaded)
        self.cache.give_back(id, activations)
            
def shuffle(cache_dir):
    cache = Cache(cache_dir)
    shuffler = Shuffler(cache)
    shuffler.continuous_shuffle()