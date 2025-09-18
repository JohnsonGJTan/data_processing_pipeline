from pathlib import Path

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