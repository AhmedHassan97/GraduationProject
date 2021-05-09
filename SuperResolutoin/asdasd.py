from itertools import chain

list2 = [[1,2,2],[1,2,2],[1,2,2]]
flatten_list = list(chain.from_iterable(list2))

print(flatten_list)
