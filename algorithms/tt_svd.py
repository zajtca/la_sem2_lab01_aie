# algorithms/tt_svd.py

"""
TT-SVD алгоритм: разложение плотного тензора в TT-формат.
"""

import math

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def tt_svd(
    tensor: DenseTensor,
    backend: BackendInterface,
    max_rank: int | None = None,
    eps: float = 1e-10
) -> TTTensor:
    """
    Возвращает TTTensor — тензор в TT-формате.

    Args:
        tensor:   DenseTensor с shape (n_0, n_1, ..., n_{d-1})
        backend:  интерфейс backend
        max_rank: максимальный TT-ранг (None = без ограничения)
        eps:      относительная точность усечения
    """
    shape = backend.shape(tensor)
    d = len(shape)

    if d == 1:
        core = backend.reshape(tensor, (1, shape[0], 1))
        return TTTensor([core])

    norm_a = backend.norm(tensor)
    if norm_a > 1e-30:
        delta = eps * norm_a / math.sqrt(d - 1)
    else:
        delta = 0.0

    cores: list[DenseTensor] = []
    c = tensor
    r_prev = 1

    for k in range(d - 1):
        n_k = shape[k]
        total = backend.size(c)
        rows = r_prev * n_k
        cols = total // rows

        mat = backend.reshape(c, (rows, cols))
        u, s, vt = backend.svd(mat, full_matrices=False)

        rank = _compute_truncated_rank(s, delta, max_rank)

        u_trunc = _truncate_columns(u, rank, backend)
        core = backend.reshape(u_trunc, (r_prev, n_k, rank))
        cores.append(core)

        s_trunc = _truncate_vector(s, rank, backend)
        vt_trunc = _truncate_rows(vt, rank, backend)
        c = _multiply_diag_matrix(s_trunc, vt_trunc, rank, backend)

        r_prev = rank

    last_core = backend.reshape(c, (r_prev, shape[d - 1], 1))
    cores.append(last_core)

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _compute_truncated_rank(
    S: DenseTensor,
    delta: float,
    max_rank: int | None
) -> int:
    """
    Возвращает ранг усечения по сингулярным значениям.

    Args:
        S:        DenseTensor (k,) — сингулярные значения по убыванию
        delta:    порог усечения
        max_rank: максимальный ранг (None = без ограничения)
    """
    k = S.shape[0]
    if k == 0:
        return 1

    sigma = S.data

    sigma_max = sigma[0]
    threshold = max(1e-12, 1e-8 * sigma_max)
    r_hat = 0
    for j in range(k):
        if sigma[j] > threshold:
            r_hat += 1
        else:
            break
    if r_hat < 1:
        r_hat = 1

    rank = r_hat
    if delta > 0.0:
        tail = 0.0
        j = r_hat
        while j > 1:
            candidate = tail + sigma[j - 1] * sigma[j - 1]
            if candidate <= delta * delta:
                tail = candidate
                rank = j - 1
                j -= 1
            else:
                break

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

    Используется после SVD для усечения матрицы левых сингулярных векторов:
        U in R^{m x n} -> U_trunc in R^{m x rank}

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
