import numpy as np


def bitpaircount_u1(a, b):
    assert a.dtype == "u1"
    assert b.dtype == "u1"
    unpack_a = (
        a.map_blocks(np.unpackbits, drop_axis=0).compute().astype("u1")
    )  # compute needed for correct shape
    unpack_b = b.map_blocks(np.unpackbits, drop_axis=0).compute().astype("u1")
    index = ((unpack_a << 1) | unpack_b).reshape(-1, 8)

    selection = np.array([0, 1, 2, 3], dtype="u1")
    print("selection")
    sel = np.where((index[..., np.newaxis]) == selection, True, False)
    print("summing stuff")
    to_return = sel.sum(axis=0).reshape(8, 2, 2)
    print("finished summing stuff")
    return to_return


def bitpaircount(a, b):
    assert a.dtype.kind == "u"
    assert b.dtype.kind == "u"
    nbytes = max(a.dtype.itemsize, b.dtype.itemsize)

    a, b = np.broadcast_arrays(a, b)

    bytewise_counts = []
    for i in range(nbytes):
        s = (nbytes - 1 - i) * 8
        bitc = bitpaircount_u1((a >> s).astype("u1"), (b >> s).astype("u1"))
        bytewise_counts.append(bitc)
    return np.concatenate(bytewise_counts, axis=0)


def mutual_information(a, b, base=2):
    size = np.prod(np.broadcast_shapes(a.shape, b.shape))
    print("run bitpaircount")
    counts = bitpaircount(a, b)
    print("finished bitpaircount")
    print(size)

    p = counts.astype("float") / size
    pr = p.sum(axis=-1)[..., np.newaxis]
    ps = p.sum(axis=-2)[..., np.newaxis, :]

    return np.where(p > 0, p * np.log(p / (pr * ps)), 0).sum(axis=(-1, -2)) / np.log(
        base
    )


def bitinformation(a, axis=0):
    sa = tuple(slice(0, -1) if i == axis else slice(None) for i in range(len(a.shape)))
    sb = tuple(
        slice(1, None) if i == axis else slice(None) for i in range(len(a.shape))
    )
    return mutual_information(a[sa], a[sb])
