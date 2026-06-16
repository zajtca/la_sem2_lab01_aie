# algorithms/tensor_operations.py

"""
Базовые операции с TT-тензорами.

Все операции работают напрямую с TT-ядрами,
не восстанавливая полный тензор.

Содержит:
    - tt_add:         поэлементное сложение
    - tt_scalar_mul:  умножение на скаляр
    - tt_hadamard:    поэлементное произведение (Адамар)
    - tt_dot:         скалярное произведение <A, B>
    - tt_norm:        Фробениусова норма
    - tt_diff_norm:   ||A - B||_F без восстановления полных тензоров

Все операции через backend.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


Number = int | float


def tt_add(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного сложения двух TT-тензоров.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    d = len(tt1.cores)

    if d == 1:
        a = tt1.cores[0]
        b = tt2.cores[0]
        n = a.shape[1]
        data = [a.data[i] + b.data[i] for i in range(n)]
        return TTTensor([DenseTensor((1, n, 1), data=data)])

    cores = []
    for k in range(d):
        a_core = tt1.cores[k]
        b_core = tt2.cores[k]
        ra_l, n, ra_r = a_core.shape
        rb_l, _, rb_r = b_core.shape
        a = a_core.data
        b = b_core.data

        if k == 0:
            new_l = 1
            new_r = ra_r + rb_r
            row_off = 0
            col_off = ra_r
        elif k == d - 1:
            new_l = ra_l + rb_l
            new_r = 1
            row_off = ra_l
            col_off = 0
        else:
            new_l = ra_l + rb_l
            new_r = ra_r + rb_r
            row_off = ra_l
            col_off = ra_r

        data = [0.0] * (new_l * n * new_r)

        for p in range(ra_l):
            for i in range(n):
                src_base = p * (n * ra_r) + i * ra_r
                dst_base = p * (n * new_r) + i * new_r
                for c in range(ra_r):
                    data[dst_base + c] = a[src_base + c]

        for q in range(rb_l):
            for i in range(n):
                src_base = q * (n * rb_r) + i * rb_r
                dst_base = (q + row_off) * (n * new_r) + i * new_r + col_off
                for c in range(rb_r):
                    data[dst_base + c] = b[src_base + c]

        cores.append(DenseTensor((new_l, n, new_r), data=data))

    return TTTensor(cores)


def tt_scalar_mul(
    tt: TTTensor,
    alpha: Number,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат умножения TT-тензора на скаляр.
    Модифицируем только первое ядро.

    Args:
        tt:      TTTensor
        alpha:   число
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    first = cores[0]
    scaled = DenseTensor(first.shape, data=[alpha * x for x in first.data])
    cores[0] = scaled
    return TTTensor(cores)


def tt_hadamard(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> TTTensor:
    """
    Возвращает результат поэлементного произведения (произведения Адамара).

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    d = len(tt1.cores)
    cores = []

    for k in range(d):
        a_core = tt1.cores[k]
        b_core = tt2.cores[k]
        ra_l, n, ra_r = a_core.shape
        rb_l, _, rb_r = b_core.shape
        a = a_core.data
        b = b_core.data

        new_l = ra_l * rb_l
        new_r = ra_r * rb_r
        data = [0.0] * (new_l * n * new_r)

        for i in range(n):
            for p in range(ra_l):
                for s in range(ra_r):
                    av = a[p * (n * ra_r) + i * ra_r + s]
                    for q in range(rb_l):
                        for t in range(rb_r):
                            bv = b[q * (n * rb_r) + i * rb_r + t]
                            row = p * rb_l + q
                            col = s * rb_r + t
                            data[row * (n * new_r) + i * new_r + col] = av * bv

        cores.append(DenseTensor((new_l, n, new_r), data=data))

    return TTTensor(cores)


def tt_dot(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> Number:
    """
    Возвращает скалярное произведение двух TT-тензоров: <tt1, tt2>.

    Args:
        tt1, tt2: TTTensor с одинаковым shape
        backend:  интерфейс backend
    """
    d = len(tt1.cores)
    z = None

    for k in range(d):
        a_core = tt1.cores[k]
        b_core = tt2.cores[k]
        ra_l, n, ra_r = a_core.shape
        rb_l, _, rb_r = b_core.shape
        a = a_core.data
        b = b_core.data

        acc = [0.0] * (ra_r * rb_r)

        for i in range(n):
            a_i_data = [0.0] * (ra_l * ra_r)
            for p in range(ra_l):
                src_base = p * (n * ra_r) + i * ra_r
                dst_base = p * ra_r
                for s in range(ra_r):
                    a_i_data[dst_base + s] = a[src_base + s]

            b_i_data = [0.0] * (rb_l * rb_r)
            for q in range(rb_l):
                src_base = q * (n * rb_r) + i * rb_r
                dst_base = q * rb_r
                for t in range(rb_r):
                    b_i_data[dst_base + t] = b[src_base + t]

            a_i = DenseTensor((ra_l, ra_r), data=a_i_data)
            b_i = DenseTensor((rb_l, rb_r), data=b_i_data)
            at_i = backend.transpose(a_i)

            if z is None:
                contrib = backend.matmul(at_i, b_i)
            else:
                tmp = backend.matmul(at_i, z)
                contrib = backend.matmul(tmp, b_i)

            cd = contrib.data
            for idx in range(ra_r * rb_r):
                acc[idx] += cd[idx]

        z = DenseTensor((ra_r, rb_r), data=acc)

    return z.data[0]


def tt_norm(
    tt: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает Фробениусову норму TT-тензора.

    Args:
        tt:      TTTensor
        backend: интерфейс backend
    """
    value = tt_dot(tt, tt, backend)
    return math.sqrt(max(value, 0.0))


def tt_diff_norm(
    tt1: TTTensor,
    tt2: TTTensor,
    backend: BackendInterface
) -> float:
    """
    Возвращает норму разности: ||tt1 - tt2||_F.
    Вычисляется без восстановления полных тензоров:

    Args:
        tt1, tt2: TTTensor
        backend:  интерфейс backend
    """
    aa = tt_dot(tt1, tt1, backend)
    ab = tt_dot(tt1, tt2, backend)
    bb = tt_dot(tt2, tt2, backend)
    value = aa - 2.0 * ab + bb
    return math.sqrt(max(value, 0.0))
