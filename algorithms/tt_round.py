# algorithms/tt_round.py

"""
TT-округление.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface
from algorithms.canonical_form import right_canonicalize


def tt_round(
    tt: TTTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор с уменьшенными рангами

    Args:
        tt:       исходный тензор
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    d = len(tt.cores)
    if d == 1:
        return tt.copy()

    rc = right_canonicalize(tt, backend)
    cores = [core.copy() for core in rc.cores]

    g1_norm = cores[0].norm()
    if g1_norm > 1e-30:
        delta = eps * g1_norm / math.sqrt(d - 1)
    else:
        delta = 0.0

    for k in range(d - 1):
        core = cores[k]
        r_left, n_k, r_right = core.shape

        mat = backend.reshape(core, (r_left * n_k, r_right))
        u, s, vt = backend.svd(mat, full_matrices=False)

        rank = _compute_rank(s, delta, max_rank)

        u_trunc = _truncate_columns(u, rank, backend)
        cores[k] = backend.reshape(u_trunc, (r_left, n_k, rank))

        s_trunc = _truncate_vector(s, rank, backend)
        vt_trunc = _truncate_rows(vt, rank, backend)
        carry = _multiply_diag_matrix(s_trunc, vt_trunc, rank, backend)

        nxt = cores[k + 1]
        n_left, n_next, n_right = nxt.shape
        nxt_mat = backend.reshape(nxt, (n_left, n_next * n_right))
        new_nxt = backend.matmul(carry, nxt_mat)
        cores[k + 1] = backend.reshape(new_nxt, (rank, n_next, n_right))

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает int ранг усечения по вектору сингулярных значений.

    Args:
        S:        одномерный тензор формы (k,) — сингулярные значения
                  в порядке убывания
        delta:    абсолютный порог усечения (0 — без усечения по delta)
        max_rank: максимально допустимый ранг (None = без ограничения)
    """
    k = S.shape[0]
    if k == 0:
        return 1

    sigma = S.data
    rank = k

    if delta > 0:
        threshold = delta * delta
        tail = 0.0
        r = k
        while r > 1:
            tail += sigma[r - 1] * sigma[r - 1]
            if tail <= threshold:
                r -= 1
            else:
                break
        rank = r

    if max_rank is not None:
        rank = min(rank, max_rank)

    return max(1, rank)


def _truncate_columns(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank столбцов исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (m, n)
        rank:    число сохраняемых столбцов
        backend: интерфейс backend
    """
    m, n = backend.shape(matrix)
    src = matrix.data
    data = []
    for i in range(m):
        base = i * n
        data.extend(src[base:base + rank])
    return DenseTensor((m, rank), data=data)


def _truncate_rows(
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает матрицу, составленную из первых rank строк исходной матрицы.

    Args:
        matrix:  двумерный тензор формы (k, n)
        rank:    число сохраняемых строк
        backend: интерфейс backend
    """
    _, n = backend.shape(matrix)
    data = list(matrix.data[:rank * n])
    return DenseTensor((rank, n), data=data)


def _truncate_vector(
    vector: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает вектор, состоящий из первых rank элементов исходного вектора.

    Args:
        vector:  одномерный тензор формы (k,)
        rank:    число сохраняемых элементов
        backend: интерфейс backend
    """
    data = list(vector.data[:rank])
    return DenseTensor((rank,), data=data)


def _multiply_diag_matrix(
    diag_vec: DenseTensor,
    matrix: DenseTensor,
    rank: int,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает произведение диагональной матрицы на обычную матрицу:
        diag(diag_vec) @ matrix

    Args:
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        matrix:   двумерный тензор формы (rank, n)
        rank:     число строк матрицы и длина диагонального вектора
        backend:  интерфейс backend
    """
    _, n = backend.shape(matrix)
    src = matrix.data
    dvec = diag_vec.data
    data = [0.0] * (rank * n)
    for i in range(rank):
        scale = dvec[i]
        base = i * n
        for j in range(n):
            data[base + j] = scale * src[base + j]
    return DenseTensor((rank, n), data=data)
