from pathlib import Path
from itertools import product

# prefix components:
space =  '    '
branch = '│   '
# pointers:
tee =    '├── '
last =   '└── '

def tree(dir_path: Path, prefix: str=''):
    """A recursive generator, given a directory Path object
    will yield a visual tree structure line by line
    with each line prefixed by the same characters
    """    
    contents = list(dir_path.iterdir())
    # contents each get pointers that are ├── with a final └── :
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        yield prefix + pointer + path.name
        if path.is_dir(): # extend the prefix and recurse:
            extension = branch if pointer == tee else space 
            # i.e. space because last, └── , above so no more |
            yield from tree(path, prefix=prefix+extension)

def print_dir_tree(dir_path: Path):
    print(dir_path.name)
    for line in tree(dir_path):
        print(line)

#def bin_search(val: float, bins: list[float]):
#        
#    # assume inclusions are of the form (a,b]
#
#    assert len(bins) > 0, 'empty bins'
#
#    assert bins == sorted(bins), 'bins are not sorted'
#
#    if val <= bins[0]:
#        return '(-inf, ' + str(bins[0]) + ']'
#    elif val > bins[-1]:
#        return '(' + str(bins[-1]) + ', inf)'
#    else:
#        # binary search for largest bin less than val
#        l, r = 0, len(bins) - 1
#        while l < r:
#            m = (l + r) // 2
#            if val <= bins[m]:
#                r = m
#            else: # val > bins[m]
#                l = m + 1
#        return '(' + str(bins[r-1]) + ', ' + str(bins[r]) + ']'

def configs_from_dict(d: dict) -> list[dict]:

    keys, values = zip(*d.items())
    return [dict(zip(keys, p)) for p in product(*values)]

def is_sub_with_gap(sub, lst):
    ln, j = len(sub), 0
    for ele in lst:
        if ele == sub[j]:
            j += 1
        if j == ln:
            return True
    return False

def str_to_path(path: str | Path):

    if isinstance(path, str):
        path = Path(path)

    return path