import logging
from collections import OrderedDict
from functools import wraps

logger = logging.getLogger(__name__)


class LRU(OrderedDict):
    "Limit size, evicting the least recently looked-up key when full"

    def __init__(self, maxsize=128, *args, **kwds):
        self.maxsize = maxsize
        super().__init__(*args, **kwds)

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


__cache = LRU(maxsize=10)


def _clear_cache():
    global __cache
    __cache = LRU(maxsize=10)


def cache(func):
    @wraps(func)
    def f(self, *args, **kwargs):
        # Quick hack for leaf dataset: remove attributes with _ in front of it
        _dict = {k: v for (k, v) in self.__dict__.items() if not k.startswith("_")}
        key = hash(str(_dict))
        global __cache
        if key in __cache:
            logger.info(f"Cache Hit in {f.__name__}!")
            return __cache[key]
        else:
            logger.info(f"Cache Miss in {f.__name__}!")
            __cache[key] = tmp = func(self, *args, **kwargs)
            return tmp

    return f
