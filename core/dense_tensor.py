# core/dense_tensor.py

"""Функции для работы с тензорами в стандартной плотной форме."""


from __future__ import annotations

import random
import math

from core.utils import (
    validate_shape,
    compute_size,
    compute_strides,
    multi_index_to_flat,
    flat_to_multi_index,
check_shapes_match,
)


class DenseTensor:
    """
    Плотный тензор произвольного порядка.

    Атрибуты:
        shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        ndim:    порядок тензора (число мод)
        size:    общее число элементов
        data:    плоский список значений (row-major / C-order)
        strides: шаги для перевода мультииндекса в плоский индекс
    """

    __slots__ = ('shape', 'ndim', 'size', 'data', 'strides')

    # ────────────────────────────────────────────
    # Конструкторы
    # ────────────────────────────────────────────

    def __init__(
        self,
        shape: tuple[int, ...] | list[int],
        data: list[float] | None = None,
        fill: float = 0.0
    ) -> None:
        """
        Создаёт тензор заданной формы.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            data:  плоский список значений (если None — заполняется fill)
            fill:  значение для заполнения (по умолчанию 0.0)
        """
        shape = validate_shape(shape)
        size = compute_size(shape)

        if data is None:
            data = [fill] * size
        elif len(data) != size:
            raise ValueError(
                f"Длина data ({len(data)}) не соответствует размеру "
                f"тензора ({size}) для формы {shape}"
            )

        self.shape = shape
        self.ndim = len(shape)
        self.size = size
        self.data = data
        self.strides = compute_strides(shape)

    @staticmethod
    def zeros(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный нулями.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=0.0)

    @staticmethod
    def ones(shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает тензор, заполненный единицами.

        Args:
            shape: кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
        """
        return DenseTensor(shape, fill=1.0)

    @staticmethod
    def random(
        shape: tuple[int, ...] | list[int],
        low: int = -5,
        high: int = 5,
        integer: bool = True,
        seed: int | None = None
    ) -> DenseTensor:
        """
        Возвращает тензор со случайными значениями.

        Args:
            shape:   кортеж размеров по каждой моде (n_0, n_1, ..., n_{d-1})
            low:     нижняя граница значений тензора
            high:    верхняя граница значений тензора
            integer: True — целые числа, False — вещественные
            seed:    seed для воспроизводимости (None — без фиксации)

        NB: эта функция не тестируется, ее можно использовать для отладки
        """
        shape = validate_shape(shape)
        size = compute_size(shape)
        rng = random.Random(seed)

        if integer:
            data = [float(rng.randint(low, high)) for _ in range(size)]
        else:
            data = [rng.uniform(low, high) for _ in range(size)]

        return DenseTensor(shape, data=data)

    @staticmethod
    def from_nested_list(nested: list) -> DenseTensor:
        """
        Создаёт тензор из вложенного списка Python.
        Автоматически определяет shape.

        Args:
            nested: список
        """
        shape: list[int] = []
        cursor = nested
        while isinstance(cursor, (list, tuple)):
            shape.append(len(cursor))
            cursor = cursor[0] if len(cursor) > 0 else None

        data: list[float] = []

        def _flatten(node) -> None:
            if isinstance(node, (list, tuple)):
                for item in node:
                    _flatten(item)
            else:
                data.append(float(node))

        _flatten(nested)
        return DenseTensor(tuple(shape), data=data)

    # ────────────────────────────────────────────
    # Индексация
    # ────────────────────────────────────────────

    def _validate_index(
        self,
        multi_index: tuple[int, ...] | int
    ) -> tuple[int, ...]:
        """
        Возвращает нормализованный мультииндекс в виде кортежа.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        if isinstance(multi_index, int):
            multi_index = (multi_index,)
        else:
            multi_index = tuple(multi_index)

        if len(multi_index) != self.ndim:
            raise IndexError(
                f"Ожидается {self.ndim} индексов, получено {len(multi_index)}"
            )

        normalized: list[int] = []
        for k, idx in enumerate(multi_index):
            if idx < 0:
                idx += self.shape[k]
            if idx < 0 or idx >= self.shape[k]:
                raise IndexError(
                    f"Индекс {idx} вне диапазона для моды {k} "
                    f"(размер {self.shape[k]})"
                )
            normalized.append(idx)

        return tuple(normalized)

    def __getitem__(self, multi_index: tuple[int, ...] | int) -> float:
        """
        Возвращает значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
        """
        idx = self._validate_index(multi_index)
        flat = multi_index_to_flat(idx, self.strides)
        return self.data[flat]

    def __setitem__(
        self,
        multi_index: tuple[int, ...] | int,
        value: float
    ) -> None:
        """
        Устанавливает новое значение элемента по заданному мультииндексу.

        Args:
            multi_index: кортеж индексов (i_0, i_1, ..., i_{d-1}) или целое число
            value:       новое значение (число)
        """
        idx = self._validate_index(multi_index)
        flat = multi_index_to_flat(idx, self.strides)
        self.data[flat] = value

    # ────────────────────────────────────────────
    # Преобразования формы
    # ────────────────────────────────────────────

    def reshape(self, new_shape: tuple[int, ...] | list[int]) -> DenseTensor:
        """
        Возвращает новый объект тензора с новой формой и скопированными данными.

        Args:
            new_shape: кортеж новых размеров (n'_0, n'_1, ..., n'_{k-1})
        """
        new_shape = validate_shape(new_shape)
        if compute_size(new_shape) != self.size:
            raise ValueError(
                f"Нельзя сделать reshape из {self.shape} в {new_shape}: "
                f"разное число элементов"
            )
        return DenseTensor(new_shape, data=self.data[:])

    def unfolding(self, mode: int) -> DenseTensor:
        """
        Возвращает матрицу — развертку тензора по моде n.

        Args:
            mode: номер моды (0 ≤ mode < ndim), которая становится индексом строк
        """
        if mode < 0 or mode >= self.ndim:
            raise ValueError(
                f"mode={mode} вне диапазона [0, {self.ndim - 1}]"
            )

        n_mode = self.shape[mode]
        cols = self.size // n_mode
        result = DenseTensor.zeros((n_mode, cols))
        r_data = result.data
        src = self.data
        shape = self.shape
        d = self.ndim

        for flat in range(self.size):
            multi = flat_to_multi_index(flat, shape)
            row = multi[mode]
            col = 0
            for a in range(d):
                if a == mode:
                    continue
                col = col * shape[a] + multi[a]
            r_data[row * cols + col] = src[flat]

        return result

    def left_unfolding(self, k: int) -> DenseTensor:
        """
        Возвращает матрицу — "левую развертку" тензора для TT-SVD.

        Args:
            k: номер границы разбиения (0 ≤ k < ndim - 1)
        """
        if k < 0 or k >= self.ndim:
            raise ValueError(
                f"k={k} вне диапазона [0, {self.ndim - 1}]"
            )

        rows = 1
        for a in range(k + 1):
            rows *= self.shape[a]
        cols = self.size // rows
        return self.reshape((rows, cols))

    # ────────────────────────────────────────────
    # Копирование
    # ────────────────────────────────────────────

    def copy(self) -> DenseTensor:
        """Возвращает глубокую копию тензора."""
        return DenseTensor(self.shape, data=self.data[:])

    # ────────────────────────────────────────────
    # Арифметика
    # ────────────────────────────────────────────

    def norm(self) -> float:
        """Возвращает Фробениусову норму тензора."""
        return math.sqrt(sum(x * x for x in self.data))

    def __add__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного сложения: t1 + t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        data = [self.data[i] + other.data[i] for i in range(self.size)]
        return DenseTensor(self.shape, data=data)

    def __sub__(self, other: DenseTensor) -> DenseTensor:
        """
        Возвращает тензор — результат поэлементного вычитания: t1 - t2.

        Args:
            other: t2
        """
        check_shapes_match(self.shape, other.shape)
        data = [self.data[i] - other.data[i] for i in range(self.size)]
        return DenseTensor(self.shape, data=data)

    def __mul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: t1 * scalar.

        Args:
            scalar: число
        """
        data = [x * scalar for x in self.data]
        return DenseTensor(self.shape, data=data)

    def __rmul__(self, scalar: float | int) -> DenseTensor:
        """
        Возвращает тензор — результат умножения тензора на скаляр: scalar * t1.

        Args:
            scalar: число, на которое умножаем
        """
        return self.__mul__(scalar)

    def __neg__(self) -> DenseTensor:
        """Возвращает тензор — результат умножения тензора на -1."""
        return self.__mul__(-1.0)

    # ────────────────────────────────────────────
    # Сравнение и отладка
    # ────────────────────────────────────────────

    def allclose(
        self,
        other: DenseTensor,
        atol: float = 1e-8,
        rtol: float = 1e-5
    ) -> bool:
        """
        Возвращает True, если тензоры равны с заданной точностью.

        Условие равенства: shape равны и для каждой пары элементов
        тензоров с равными индексами выполняется:
            |a - b| <= atol + rtol * max(|a|, |b|)


        Args:
            other: DenseTensor для сравнения
            atol:  абсолютная погрешность (по умолчанию 1e-8)
            rtol:  относительная погрешность (по умолчанию 1e-5)
        """
        if tuple(self.shape) != tuple(other.shape):
            return False
        for a, b in zip(self.data, other.data):
            if abs(a - b) > atol + rtol * max(abs(a), abs(b)):
                return False
        return True

    def to_nested_list(self) -> list:
        """Возвращает тензор в формате вложенного списка."""
        if self.ndim == 0:
            return self.data[0]

        def _build(offset: int, dim: int):
            if dim == self.ndim - 1:
                return [self.data[offset + i] for i in range(self.shape[dim])]
            stride = self.strides[dim]
            return [
                _build(offset + i * stride, dim + 1)
                for i in range(self.shape[dim])
            ]

        return _build(0, 0)

    def __repr__(self) -> str:
        """
        Возвращает строковое представление тензора для отладки.

        NB: эта функция не проверяется тестами, ее реализация может быть произвольной
        """
        return f"DenseTensor(shape={self.shape}, data={self.data})"

    def __str__(self) -> str:
        """Возвращает строковое представление тензора для отладки."""
        return self.__repr__()
