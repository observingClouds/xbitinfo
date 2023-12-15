import dask.array as da
import numpy as np
import numpy.ma as nm


def exponent_bias(dtype):
    """
    Returns the exponent bias for a given floating-point dtype.

    Example
    -------
    >>> exponent_bias("f4")
    127
    >>> exponent_bias("f8")
    1023
    """
    info = np.finfo(dtype)
    exponent_bits = info.bits - info.nmant - 1
    return 2 ** (exponent_bits - 1) - 1


def exponent_mask(dtype):
    """
    Returns exponent mask for a given floating-point dtype.

    Example
    -------
    >>> np.binary_repr(exponent_mask(np.float32), width=32)
    '01111111100000000000000000000000'
    >>> np.binary_repr(exponent_mask(np.float16), width=16)
    '0111110000000000'
    """
    if dtype == np.float16:
        mask = 0x7C00
    elif dtype == np.float32:
        mask = 0x7F80_0000
    elif dtype == np.float64:
        mask = 0x7FF0_0000_0000_0000
    return mask


def signed_exponent(A):
    """
    Transform biased exponent notation to signed exponent notation.

    Parameters
    ----------
    A : :py:class:`numpy.array`
        Array to transform

    Returns
    -------
    B : :py:class:`numpy.array`

    Example
    -------
    >>> A = np.array(0.03125, dtype="float32")
    >>> np.binary_repr(A.view("uint32"), width=32)
    '00111101000000000000000000000000'
    >>> np.binary_repr(signed_exponent(A), width=32)
    '01000010100000000000000000000000'
    >>> A = np.array(0.03125, dtype="float64")
    >>> np.binary_repr(A.view("uint64"), width=64)
    '0011111110100000000000000000000000000000000000000000000000000000'
    >>> np.binary_repr(signed_exponent(A), width=64)
    '0100000001010000000000000000000000000000000000000000000000000000'
    """
    itemsize = A.dtype.itemsize
    uinttype = f"u{itemsize}"
    inttype = f"i{itemsize}"

    sign_mask = 1 << np.finfo(A.dtype).bits - 1
    sfmask = sign_mask | (1 << np.finfo(A.dtype).nmant) - 1
    emask = exponent_mask(A.dtype)
    esignmask = sign_mask >> 1

    sbits = np.finfo(A.dtype).nmant
    if itemsize == 8:
        sbits = np.uint64(sbits)
        emask = np.uint64(emask)
    bias = exponent_bias(A.dtype)

    ui = A.view(uinttype)
    sf = ui & sfmask
    e = ((ui & emask) >> sbits).astype(inttype) - bias
    max_eabs = np.iinfo(A.view(uinttype).dtype).max >> sbits
    eabs = abs(e) % (max_eabs + 1)
    esign = np.where(e < 0, esignmask, 0)
    if itemsize == 8:
        eabs = np.uint64(eabs)
        esign = np.uint64(esign)
    esigned = esign | (eabs << sbits)
    B = (sf | esigned).view(np.int64)
    return B


def bitpaircount_u1(a, b):
    assert a.dtype == "u1"
    assert b.dtype == "u1"
    unpack_a = (
        a.flatten()
        .map_blocks(
            np.unpackbits,
            drop_axis=0,
            meta=np.array((), dtype=np.uint8),
            chunks=(a.size * 8,),
        )
        .astype("u1")
    )
    unpack_b = (
        b.flatten()
        .map_blocks(
            np.unpackbits,
            drop_axis=0,
            meta=np.array((), dtype=np.uint8),
            chunks=(b.size * 8,),
        )
        .astype("u1")
    )
    index = ((unpack_a << 1) | unpack_b).reshape(-1, 8)

    selection = np.array([0, 1, 2, 3], dtype="u1")
    sel = np.where((index[..., np.newaxis]) == selection, True, False)
    to_return = sel.sum(axis=0).reshape(8, 2, 2)
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
    counts = bitpaircount(a, b)

    p = counts.astype("float") / size
    p = da.ma.masked_equal(p, 0)
    pr = p.sum(axis=-1)[..., np.newaxis]
    ps = p.sum(axis=-2)[..., np.newaxis, :]
    mutual_info = (p * np.ma.log(p / (pr * ps))).sum(axis=(-1, -2)) / np.log(base)
    return mutual_info


def bitinformation(a, axis=0):
    sa = tuple(slice(0, -1) if i == axis else slice(None) for i in range(len(a.shape)))
    sb = tuple(
        slice(1, None) if i == axis else slice(None) for i in range(len(a.shape))
    )
    return mutual_information(a[sa], a[sb])
