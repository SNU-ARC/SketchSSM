# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ReplaySSM Kimi Delta Attention (KDA) decode kernels (Gluon/Triton).

A W=16 window over the FP32 native page S0 is kept as exact parallel WY
factors, delta_s = u_s - S0 @ pi_s, so that
S_t = S0 * d_t + sum_s delta_s ell_s(t)^T. Non-flush steps read S0 once for
the current query and key and never write it; the flush step (pos 15) folds
the window into the native page in place. Raw BF16 k/v and FP32 gate/beta are
kept for the partial flush that precedes a prefill or an eviction.

Ring slots are owned by physical pages: ``Owners[slot]`` is the page whose
pending window the slot holds. Page 0 and negative IDs are padding.
"""

from vllm.triton_utils import gl, gluon, tl, triton

KDA_REPLAY_WINDOW = 16
KDA_REPLAY_HEAD_DIM = 128
KDA_REPLAY_LOWER_BOUND = -5.0


@gluon.jit
def replay_step(
    Q,
    K,
    V,
    Gate,
    Beta,
    A,
    Bias,
    Slots,
    Pos,
    KR,
    VR,
    GR,
    BR,
    PrefixR,
    DR,
    DeltaR,
    QueryScaled,
    KeyScaled,
    Rhs,
    ReplayOut,
    Scalars,
    H: gl.constexpr,
):
    head, row = gl.program_id(0), gl.program_id(1)
    slot = gl.load(Slots + row)
    if slot < 0:
        return
    pos = gl.load(Pos + slot)
    # One warp keeps the ring reductions in registers; vectorize along K/V.
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [1, 1], [1, 0])
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    t = gl.arange(0, 16, layout=gl.SliceLayout(1, layout))
    offset = (row * H + head) * 128
    raw_k = gl.load(K + offset + k).to(gl.float32)
    value = gl.load(V + offset + k).to(gl.float32)
    query = gl.load(Q + offset + k).to(gl.float32)
    query *= gl.rsqrt(gl.sum(query * query) + 1e-6) * 128**-0.5
    key = raw_k * gl.rsqrt(gl.sum(raw_k * raw_k) + 1e-6)
    raw_g = gl.load(Gate + offset + k).to(gl.float32)
    raw_g += gl.load(Bias + head * 128 + k)
    log_a = -5.0 / (1.0 + gl.exp(-gl.exp(gl.load(A + head)) * raw_g))
    beta = 1.0 / (1.0 + gl.exp(-gl.load(Beta + row * H + head).to(gl.float32)))
    base = (slot * H + head) * 16
    gl.store(KR + (base + pos) * 128 + k, raw_k)
    gl.store(VR + (base + pos) * 128 + k, value)
    gl.store(GR + (base + pos) * 128 + k, log_a)
    gl.store(BR + base + pos, beta)
    previous = gl.load(PrefixR + (slot * H + head) * 128 + k, mask=pos > 0, other=0.0)
    prefix = previous + log_a
    decay = gl.exp(prefix)
    gl.store(PrefixR + (slot * H + head) * 128 + k, prefix)
    gl.store(DR + (base + pos) * 128 + k, key * gl.exp(-prefix))
    past_d = gl.load(
        DR + (base + t[:, None]) * 128 + k[None, :], mask=t[:, None] < pos, other=0.0
    )
    ell = past_d * decay[None, :]
    kk_inner = gl.sum(ell * key[None, :], axis=1)
    kq_inner = gl.sum(ell * query[None, :], axis=1)
    delta = gl.load(
        DeltaR + (base + t[:, None]) * 128 + k[None, :],
        mask=t[:, None] < pos,
        other=0.0,
    )
    rhs = beta * (value - gl.sum(delta * kk_inner[:, None], axis=0))
    ring_out = gl.sum(delta * kq_inner[:, None], axis=0)
    gl.store(QueryScaled + offset + k, decay * query)
    gl.store(KeyScaled + offset + k, decay * key)
    gl.store(Rhs + offset + k, rhs)
    gl.store(ReplayOut + offset + k, ring_out)
    gl.store(Scalars + (row * H + head) * 2, beta)
    gl.store(Scalars + (row * H + head) * 2 + 1, gl.sum(key * query))


@gluon.jit
def replay_read(
    IDs,
    Slots,
    Pos,
    State,
    DeltaR,
    QueryScaled,
    KeyScaled,
    Rhs,
    ReplayOut,
    Scalars,
    Out,
    H: gl.constexpr,
    BV: gl.constexpr,
    S0: gl.constexpr,
    S1: gl.constexpr,
    S2: gl.constexpr,
    S3: gl.constexpr,
):
    block, head, row = gl.program_id(0), gl.program_id(1), gl.program_id(2)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [1, 32], [4, 1], [1, 0])
    packed: gl.constexpr = gl.BlockedLayout([1], [32], [4], [0])
    kp = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    vp = block * BV + gl.arange(0, BV, layout=packed)
    vv = block * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    slot = gl.load(Slots + row)
    offset = (row * H + head) * 128
    if slot < 0:
        gl.store(Out + offset + vp, 0.0)
        return
    pos = gl.load(Pos + slot)
    if pos == 15:
        return
    q = gl.load(QueryScaled + offset + kp)
    k = gl.load(KeyScaled + offset + kp)
    rhs = gl.load(Rhs + offset + vp)
    replay_out = gl.load(ReplayOut + offset + vp)
    beta = gl.load(Scalars + (row * H + head) * 2)
    kq = gl.load(Scalars + (row * H + head) * 2 + 1)
    physical = gl.load(IDs + row).to(gl.int64)
    state = gl.load(
        State + physical * S0 + head * S1 + vv[:, None] * S2 + kp[None, :] * S3,
        cache_modifier=".cg",
    )
    projection_k = gl.convert_layout(gl.sum(state * k[None, :], axis=1), packed)
    projection_q = gl.convert_layout(gl.sum(state * q[None, :], axis=1), packed)
    delta = rhs - beta * projection_k
    base = (slot * H + head) * 16 + pos
    gl.store(DeltaR + base * 128 + vp, delta)
    gl.store(Out + offset + vp, projection_q + replay_out + delta * kq)


@triton.jit
def replay_flush(
    IDs,
    Q,
    Slots,
    State,
    PrefixR,
    DR,
    KeyScaled,
    Rhs,
    Scalars,
    DeltaR,
    Out,
    WorkRows,
    WorkCounts,
    Capacity: tl.constexpr,
    H: tl.constexpr,
    BV: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
):
    """Fuse exact WY window update and full-state output for flush rows only."""
    tiles: tl.constexpr = 128 // BV
    total = tl.load(WorkCounts + 1) * H * tiles
    for item in range(tl.program_id(0), total, tl.num_programs(0)):
        row = tl.load(WorkRows + Capacity + item // (H * tiles))
        head = item // tiles % H
        v_start = item % tiles * BV
        slot = tl.load(Slots + row).to(tl.int64)
        k = tl.arange(0, 128)
        v = v_start + tl.arange(0, BV)
        physical = tl.load(IDs + row).to(tl.int64)
        sp = State + physical * S0 + head * S1 + v[:, None] * S2 + k[None, :] * S3
        state = tl.load(sp, cache_modifier=".cg")
        base = (slot * H + head) * 16
        prefix = tl.load(PrefixR + (slot * H + head) * 128 + k)
        decay = tl.exp(prefix)
        last_key = tl.load(KeyScaled + (row * H + head) * 128 + k)
        last_rhs = tl.load(Rhs + (row * H + head) * 128 + v)
        beta = tl.load(Scalars + (row * H + head) * 2)
        last_delta = last_rhs - beta * tl.sum(state * last_key[None, :], axis=1)
        state *= decay[None, :]
        for i in tl.static_range(16):
            left = tl.load(DR + (base + i) * 128 + k) * decay
            if i == 15:  # noqa: SIM108 -- static branch avoids slot-15 loads
                delta = last_delta
            else:
                delta = tl.load(DeltaR + (base + i) * 128 + v)
            state += delta[:, None] * left[None, :]
        tl.store(sp, state)
        query = tl.load(Q + (row.to(tl.int64) * H + head) * 128 + k).to(tl.float32)
        query *= tl.rsqrt(tl.sum(query * query) + 1.0e-6) * 128**-0.5
        tl.store(
            Out + (row.to(tl.int64) * H + head) * 128 + v,
            tl.sum(state * query[None, :], axis=1),
        )


@triton.jit
def replay_resolve(
    IDs,
    Owners,
    Slots,
    Old,
    Fresh,
    B: tl.constexpr,
    P: tl.constexpr,
    WB: tl.constexpr,
    WP: tl.constexpr,
    Pos,
    OldPos,
    Counts,
    WorkRows,
    WorkCounts,
):
    """Map rows to slots; fresh rows take empty slots first, then evict."""
    rows = tl.arange(0, WB)
    pools = tl.arange(0, WP)
    ids = tl.load(IDs + rows, rows < B, other=-1)
    owners = tl.load(Owners + pools, pools < P, other=-2)
    match = (ids[:, None] == owners[None, :]) & (ids[:, None] > 0)
    found = tl.sum(match.to(tl.int32), 1) > 0
    available = (tl.sum(match.to(tl.int32), 0) == 0) & (pools < P)
    fresh = ~found & (ids > 0) & (rows < B)
    ordinal = tl.cumsum(fresh.to(tl.int32))
    empty = available & (owners < 0)
    occupied = available & (owners >= 0)
    free_order = tl.where(
        empty,
        tl.cumsum(empty.to(tl.int32)),
        tl.sum(empty.to(tl.int32)) + tl.cumsum(occupied.to(tl.int32)),
    )
    choose = (ordinal[:, None] == free_order[None, :]) & available[None, :]
    chosen = tl.sum(tl.where(choose, pools[None, :], 0), 1)
    existing = tl.sum(tl.where(match, pools[None, :], 0), 1)
    slots = tl.where(fresh, chosen, existing)
    slots = tl.where(ids > 0, slots, -1)
    old = tl.load(Owners + slots, (slots >= 0) & fresh, other=-1)
    tl.store(Slots + rows, slots, rows < B)
    tl.store(Old + rows, old, rows < B)
    tl.store(Fresh + rows, fresh, rows < B)

    position = tl.load(Pos + slots, (slots >= 0) & fresh & (old > 0), other=0)
    tl.store(OldPos + rows, position, rows < B)
    count = tl.sum(((slots >= 0) & (rows < B)).to(tl.int64), 0)
    tl.store(Counts, tl.load(Counts) + count)

    fresh_order = tl.cumsum(fresh.to(tl.int32))
    tl.store(WorkRows + fresh_order - 1, rows, fresh)
    current_pos = tl.load(Pos + slots, slots >= 0, other=0)
    flush = (slots >= 0) & (~fresh) & (current_pos == 15) & (rows < B)
    flush_order = tl.cumsum(flush.to(tl.int32))
    tl.store(WorkRows + P + flush_order - 1, rows, flush)
    tl.store(WorkCounts, tl.sum(fresh.to(tl.int32)))
    tl.store(WorkCounts + 1, tl.sum(flush.to(tl.int32)))


@triton.jit
def _replay_acquire_row(
    row,
    h,
    IDs,
    Slots,
    Old,
    OldPos,
    Fresh,
    Owners,
    Pos,
    State,
    KR,
    VR,
    GR,
    BR,
    H: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
):
    slot = tl.load(Slots + row)
    if slot < 0:
        return
    if not tl.load(Fresh + row):
        return
    physical = tl.load(IDs + row).to(tl.int64)
    old = tl.load(Old + row).to(tl.int64)
    k = tl.arange(0, 128)
    v = tl.arange(0, 128)
    offset = h * S1 + v[:, None] * S2 + k[None, :] * S3
    if old > 0:
        # Evict: fold the old owner's pending raw updates into its page.
        raw = tl.load(State + old * S0 + offset)
        count = tl.load(OldPos + row)
        ring = (slot * H + h) * 16
        for t in range(count):
            key = tl.load(KR + (ring + t) * 128 + k).to(tl.float32)
            key *= tl.rsqrt(tl.sum(key * key) + 1e-6)
            value = tl.load(VR + (ring + t) * 128 + v).to(tl.float32)
            decay = tl.load(GR + (ring + t) * 128 + k)
            beta = tl.load(BR + ring + t)
            raw *= tl.exp(decay[None, :])
            delta = beta * (value - tl.sum(raw * key[None, :], axis=1))
            raw += delta[:, None] * key[None, :]
        tl.store(State + old * S0 + offset, raw)
    if h == 0:
        tl.store(Owners + slot, physical)
        tl.store(Pos + slot, 0)


@triton.jit
def replay_acquire(
    IDs,
    Slots,
    Old,
    OldPos,
    Fresh,
    Owners,
    Pos,
    State,
    KR,
    VR,
    GR,
    BR,
    H: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
    WorkRows,
    WorkCounts,
):
    total = tl.load(WorkCounts) * H
    for item in range(tl.program_id(0), total, tl.num_programs(0)):
        row = tl.load(WorkRows + item // H)
        h = item % H
        _replay_acquire_row(
            row,
            h,
            IDs,
            Slots,
            Old,
            OldPos,
            Fresh,
            Owners,
            Pos,
            State,
            KR,
            VR,
            GR,
            BR,
            H,
            S0,
            S1,
            S2,
            S3,
        )


@triton.jit
def replay_bump(Slots, Pos, B: tl.constexpr, W: tl.constexpr, X: tl.constexpr):
    r = tl.arange(0, X)
    slot = tl.load(Slots + r, r < B, other=-1)
    pos = tl.load(Pos + slot, slot >= 0, other=0)
    tl.store(Pos + slot, (pos + 1) % W, slot >= 0)


@triton.jit
def replay_prefill_resolve(
    IDs,
    Initial,
    Owners,
    Slots,
    FlushSlots,
    B: tl.constexpr,
    P: tl.constexpr,
    WB: tl.constexpr,
    WP: tl.constexpr,
):
    rows, pools = tl.arange(0, WB), tl.arange(0, WP)
    ids = tl.load(IDs + rows, rows < B, other=-1)
    flags = tl.load(Initial + rows, rows < B, other=False)
    owner = tl.load(Owners + pools, pools < P, other=-2)
    match = (ids[:, None] == owner[None, :]) & (ids[:, None] > 0)
    exists = tl.sum(match.to(tl.int32), 1) > 0
    slot = tl.sum(tl.where(match, pools[None, :], 0), 1)
    tl.store(Slots + rows, tl.where(exists, slot, -1), rows < B)
    tl.store(FlushSlots + rows, tl.where(exists & flags, slot, -1), rows < B)


@triton.jit
def replay_partial_flush(
    Slots,
    Pos,
    Owners,
    State,
    KR,
    VR,
    GR,
    BR,
    H: tl.constexpr,
    BV: tl.constexpr,
    S0: tl.constexpr,
    S1: tl.constexpr,
    S2: tl.constexpr,
    S3: tl.constexpr,
):
    """Fold the pos pending raw updates of a slot into its owner's page."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    block = tl.program_id(2)
    slot = tl.load(Slots + row)
    if slot >= 0:
        count = tl.load(Pos + slot)
        if count > 0:
            kk = tl.arange(0, 128)
            vv = block * BV + tl.arange(0, BV)
            physical = tl.load(Owners + slot).to(tl.int64)
            sp = State + physical * S0 + head * S1 + vv[:, None] * S2 + kk[None, :] * S3
            state = tl.load(sp)
            base = (slot * H + head) * 16
            for t in range(count):
                k = tl.load(KR + (base + t) * 128 + kk).to(tl.float32)
                k *= tl.rsqrt(tl.sum(k * k) + 1e-6)
                v = tl.load(VR + (base + t) * 128 + vv).to(tl.float32)
                log_a = tl.load(GR + (base + t) * 128 + kk)
                beta = tl.load(BR + base + t)
                state *= tl.exp(log_a[None, :])
                delta = beta * (v - tl.sum(state * k[None, :], axis=1))
                state += delta[:, None] * k[None, :]
            tl.store(sp, state)


@triton.jit
def replay_release(Slots, Owners):
    slot = tl.load(Slots + tl.program_id(0))
    if slot >= 0:
        tl.store(Owners + slot, -1)
