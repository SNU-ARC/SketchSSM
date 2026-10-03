# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SketchSSM project
"""State traffic: the FP32 state is read and written at the flush; non-flush
steps read BF16 sketch/map rows, and dense heads BF16 rows of their full state."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Traffic:
    K: int
    V: int
    W: int = 16
    erase: bool = False

    def __post_init__(self):
        if any(not isinstance(x, int) or isinstance(x, bool) for x in (self.K, self.V, self.W)):
            raise ValueError('K, V and W must be integers')
        if min(self.K, self.V) < 1 or self.W < 2:
            raise ValueError('Positive state dimensions and W >= 2 are required')

    # Internal units are FP32-equivalent words; report() converts to bytes.
    @property
    def sketch_nonflush(self):
        return 0.5 * (self.K + self.V + (self.W if self.erase else 0))

    @property
    def state(self):
        """The FP32 state of a head, read every step by Standard decode."""
        return self.K * self.V

    @property
    def dense_nonflush(self):
        """A dense head's BF16 copy of its state, built at the flush."""
        return 0.5 * self.K * self.V

    @property
    def flush(self):
        return self.state / self.W

    def report(self, mean_nonflush):
        read = 4 * (self.flush + (self.W - 1) / self.W * mean_nonflush)
        write = 4 * self.flush
        return dict(unit='bytes_per_head_per_step', K=self.K, V=self.V, W=self.W,
                    erase=self.erase, state_dtype='float32', sketch_dtype='bfloat16',
                    coefficient_map_dtype='bfloat16', state_read=read, state_write=write,
                    state_access=read + write,
                    read_reduction=4 * self.state / read,
                    access_reduction=8 * self.state / (read + write))


@dataclass(frozen=True)
class AllocationCost:
    """Allocation weights of a sketch rank and of the dense fallback.

    A rank costs K+V+eW and dense fallback costs KV: the non-flush BF16 reads
    of a rank's sketch/map rows and of a dense head's state rows are in this
    ratio, so the dense crossover rank is where both cost the same traffic.
    Report the traffic itself with Traffic.
    """
    K: int
    V: int
    W: int = 16
    erase: bool = False

    def __post_init__(self):
        Traffic(self.K, self.V, self.W, self.erase)

    @property
    def rank(self):
        return self.K + self.V + (self.W if self.erase else 0)

    @property
    def dense(self):
        return self.K * self.V

    @property
    def crossover(self):
        return min(self.K, self.dense // self.rank)

    def budget(self, mean_rank):
        import math
        if not math.isfinite(mean_rank) or mean_rank < 1:
            raise ValueError('Mean-rank budget must be finite and at least 1')
        return self.rank * mean_rank
