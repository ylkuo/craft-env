import numpy as np


def pad_slice(array, slice_r, slice_c):
    assert len(array.shape) >= 2

    r1, r2 = slice_r
    c1, c2 = slice_c
    assert r2 > r1
    assert c2 > c1

    pr1 = max(r1, 0)
    pc1 = max(c1, 0)

    sl = array[pr1:r2, pc1:c2, :]
    slr, slc = sl.shape[:2]

    padded_sl = np.zeros((r2 - r1, c2 - c1) + array.shape[2:])
    pad_fr_r = pr1 - r1
    pad_to_r = pad_fr_r + slr
    pad_fr_c = pc1 - c1
    pad_to_c = pad_fr_c + slc

    padded_sl[pad_fr_r:pad_to_r, pad_fr_c:pad_to_c, :] = sl

    return padded_sl


class Index:
    def __init__(self):
        self.contents = dict()
        self.ordered_contents = []
        self.reverse_contents = dict()

    def __getitem__(self, item):
        if item not in self.contents:
            return None
        return self.contents[item]

    def index(self, item):
        if item not in self.contents:
            idx = len(self.contents) + 1
            self.ordered_contents.append(item)
            self.contents[item] = idx
            self.reverse_contents[idx] = item
        idx = self[item]
        assert idx != 0
        return idx

    def get(self, idx):
        if idx == 0:
            return "*invalid*"
        return self.reverse_contents[idx]

    def __len__(self):
        return len(self.contents) + 1

    def __iter__(self):
        return iter(self.ordered_contents)

