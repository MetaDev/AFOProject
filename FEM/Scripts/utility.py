import itertools

flatten = lambda lst: reduce(lambda l, i: l + flatten(i) 
                    if isinstance(i, (list, tuple)) else l + [i], lst, [])
