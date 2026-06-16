# algorithms/canonical_form.py

"""
Приведение TT-тензора в канонические формы (полная правая и
левая ортогонализация ядер).
"""

from core.tt_tensor import TTTensor
from core.dense_tensor import DenseTensor
from processor_type.interface import BackendInterface


def left_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в лево-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    d = len(cores)

    for k in range(d - 1):
        core = cores[k]
        r_left, n_k, r_right = core.shape

        mat = backend.reshape(core, (r_left * n_k, r_right))
        u, s, vt = backend.svd(mat, full_matrices=False)

        rank = backend.shape(u)[1]

        cores[k] = backend.reshape(u, (r_left, n_k, rank))

        carry = _multiply_diag_matrix(s, vt, rank, backend)

        nxt = cores[k + 1]
        n_left, n_next, n_right = nxt.shape
        nxt_mat = backend.reshape(nxt, (n_left, n_next * n_right))
        new_nxt = backend.matmul(carry, nxt_mat)
        cores[k + 1] = backend.reshape(new_nxt, (rank, n_next, n_right))

    return TTTensor(cores)


def right_canonicalize(tt: TTTensor, backend: BackendInterface) -> TTTensor:
    """
    Возвращает TTTensor — новый TT-тензор в право-канонической форме.

    Args:
        tt:      исходный тензор
        backend: интерфейс backend
    """
    cores = [core.copy() for core in tt.cores]
    d = len(cores)

    for k in range(d - 1, 0, -1):
        core = cores[k]
        r_left, n_k, r_right = core.shape

        mat = backend.reshape(core, (r_left, n_k * r_right))
        u, s, vt = backend.svd(mat, full_matrices=False)

        rank = backend.shape(u)[1]

        cores[k] = backend.reshape(vt, (rank, n_k, r_right))

        carry = _multiply_columns_by_diag(u, s, backend)

        prev = cores[k - 1]
        p_left, n_prev, p_right = prev.shape
        prev_mat = backend.reshape(prev, (p_left * n_prev, p_right))
        new_prev = backend.matmul(prev_mat, carry)
        cores[k - 1] = backend.reshape(new_prev, (p_left, n_prev, rank))

    return TTTensor(cores)


# ════════════════════════════════════════════════
# Вспомогательные функции
# ════════════════════════════════════════════════

def _numerical_rank(
    S: DenseTensor,
    rel_tol: float = 1e-8,
    abs_tol: float = 1e-12
) -> int:
    """
    Возвращает числовой ранг матрицы по вектору сингулярных значений.

    Сингулярное число \sigma_i считаем ненулевым, если:
        |\sigma_i| > max(abs_tol, rel_tol * max(\sigma_1, ..., \sigma_n))

    Args:
        S:       одномерный тензор формы (k,) — сингулярные значения
                 в порядке убывания
        rel_tol: относительный допуск (по умолчанию 1e-8)
        abs_tol: абсолютный допуск (по умолчанию 1e-12)
    """
    k = S.shape[0]
    if k == 0:
        return 1

    sigma = S.data
    threshold = max(abs_tol, rel_tol * sigma[0])

    rank = 0
    for j in range(k):
        if sigma[j] > threshold:
            rank += 1
        else:
            break

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
        rank:     длина диагонального вектора
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


def _multiply_columns_by_diag(
    matrix: DenseTensor,
    diag_vec: DenseTensor,
    backend: BackendInterface
) -> DenseTensor:
    """
    Возвращает результат произведения обычной матрицы на диагональную:
        matrix @ diag(diag_vec)

    Args:
        matrix:   двумерный тензор формы (m, n)
        diag_vec: одномерный тензор формы (rank,), содержащий диагональные элементы
        backend:  интерфейс backend
    """
    m, n = backend.shape(matrix)
    src = matrix.data
    dvec = diag_vec.data
    data = [0.0] * (m * n)
    for i in range(m):
        base = i * n
        for j in range(n):
            data[base + j] = src[base + j] * dvec[j]
    return DenseTensor((m, n), data=data)
