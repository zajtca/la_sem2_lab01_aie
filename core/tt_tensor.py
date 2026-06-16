# core/tt_tensor.py

"""
Тензор в TT-формате (Tensor Train).

TT-тензор порядка d с shape (n_0, n_1, ..., n_{d-1}) хранится как
список d ядер (cores), где k-е ядро — это 3D DenseTensor с shape:
    (r_k, n_k, r_{k+1})

Граничные условия: r_0 = r_d = 1.

TT-ранги: (r_0, r_1, ..., r_d) = (1, r_1, ..., r_{d-1}, 1).
"""

from __future__ import annotations

from core.dense_tensor import DenseTensor
from core.utils import validate_shape, compute_size, flat_to_multi_index


class TTTensor:
    """
    Тензор в TT-формате.

    Атрибуты:
        cores:  список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        order:  порядок тензора d (число мод)
        shape:  кортеж (n_0, n_1, ..., n_{d-1})
        ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d), r_0 = r_d = 1
    """

    __slots__ = ('cores', 'order', 'shape', 'ranks')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(self, cores: list[DenseTensor]) -> None:
        """
        Создаёт TT-тензор из списка ядер.

        Args:
            cores: список DenseTensor, каждый с shape (r_k, n_k, r_{k+1})
        """
        self.cores = list(cores)
        self.order = len(self.cores)
        self.shape = tuple(core.shape[1] for core in self.cores)

        ranks = [self.cores[0].shape[0]]
        for core in self.cores:
            ranks.append(core.shape[2])
        self.ranks = tuple(ranks)


    @staticmethod
    def random(shape, ranks, seed=None):
        """
        Создаёт случайный TT-тензор с заданными рангами.

        Args:
            shape:  кортеж размеров мод (n_0, ..., n_{d-1})
            ranks:  кортеж TT-рангов (r_0, r_1, ..., r_d)
                    или список внутренних рангов (r_1, ..., r_{d-1})
            seed:   seed для воспроизводимости

        NB: это отладочная функция, она не проверяется тестами
        """
        shape = validate_shape(shape)
        d = len(shape)
        ranks = list(ranks)

        if len(ranks) == d - 1:
            full_ranks = [1] + ranks + [1]
        elif len(ranks) == d + 1:
            full_ranks = ranks
        else:
            full_ranks = ranks

        cores = []
        for k in range(d):
            core_seed = None if seed is None else seed + k
            core = DenseTensor.random(
                (full_ranks[k], shape[k], full_ranks[k + 1]),
                integer=False,
                seed=core_seed,
            )
            cores.append(core)

        return TTTensor(cores)

    # ────────────────────────────────────────────
    # Доступ к элементам
    # ────────────────────────────────────────────

    def get_element(
        self,
        indices: tuple[int, ...] | list[int]
    ) -> float:
        """
        Возвращает элемент TT-тензора по его мультииндексу.

        Args:
            indices: кортеж/список длины d
        """
        indices = tuple(indices)
        vec = [1.0]

        for k in range(self.order):
            core = self.cores[k]
            r_left, n_k, r_right = core.shape
            i_k = indices[k]
            c_data = core.data
            row_stride = n_k * r_right
            offset = i_k * r_right

            new_vec = [0.0] * r_right
            for b in range(r_right):
                acc = 0.0
                base = offset + b
                for a in range(r_left):
                    acc += vec[a] * c_data[a * row_stride + base]
                new_vec[b] = acc
            vec = new_vec

        return vec[0]

    # ────────────────────────────────────────────
    # Восстановление полного тензора
    # ────────────────────────────────────────────

    def full(self) -> DenseTensor:
        """Возвращает полный DenseTensor из его TT-формата."""
        result = DenseTensor.zeros(self.shape)
        r_data = result.data
        shape = self.shape

        for flat in range(result.size):
            multi = flat_to_multi_index(flat, shape)
            r_data[flat] = self.get_element(multi)

        return result

    # ────────────────────────────────────────────
    # Информация и отладка
    # ────────────────────────────────────────────

    def core_sizes(self) -> list[tuple[int, ...]]:
        """Возвращает размеры всех ядер."""
        return [core.shape for core in self.cores]

    def total_storage(self) -> int:
        """
        Возвращает общее число элементов во всех ядрах.
        Это то, сколько памяти реально занимает TT-тензор.
        """
        return sum(core.size for core in self.cores)

    def compression_ratio(self) -> float:
        """
        Возвращает отношение числа элементов полного тензора к числу
        элементов TT-тензора. Показывает, насколько TT-формат компактнее.
        """
        full_size = compute_size(self.shape)
        return full_size / self.total_storage()

    def copy(self) -> TTTensor:
        """Возвращает глубокую копию TT-тензора."""
        return TTTensor([core.copy() for core in self.cores])

    def __repr__(self) -> str:
        """
        Возвращает строковое представление TT-тензора для отладки.

        Формирует многострочную строку с основной служебной информацией
        об объекте:
            - порядок тензора (order),
            - исходная форма (shape),
            - TT-ранги (ranks),
            - размеры TT-ядер (cores),
            - суммарный объём хранения в элементах.

        NB: это отладочная функция, которая не покрывается тестами
        """
        lines = [
            "TTTensor(",
            f"  order={self.order},",
            f"  shape={self.shape},",
            f"  ranks={self.ranks},",
            f"  cores={self.core_sizes()},",
            f"  storage={self.total_storage()}",
            ")",
        ]
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Возвращает строковое представление TT-тензора.

        Делегирует работу методу __repr__, обеспечивая единый формат
        отображения при вызове.
        """
        return self.__repr__()
