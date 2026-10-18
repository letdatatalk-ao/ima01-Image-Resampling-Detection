import numpy as np

def pcc_complex(x, y):
    """Pearson correlation coefficient for complex numbers"""
    x=np.asarray(x, dtype=complex)
    y=np.asarray(y, dtype=complex)
    xc = x - np.mean(x)
    yc = y - np.mean(y)
    scalar_product = np.sum(xc * np.conj(yc))
    norm_product = np.linalg.norm(xc) * np.linalg.norm(yc)
    return np.abs(scalar_product) / (norm_product + 1e-12)  


def patch_correlations_1d(spectrum, L, dmax):
    """Compute patch correlations for a 1D spectrum"""
    N = len(spectrum)
    n_seg=N//L
    rho = np.zeros((n_seg, dmax+1), dtype=float)

    for d in range(dmax+1):
        for i in range(n_seg):
            idx_P= i*L + np.arange(L)
            idx_Q= (i*L + d+ np.arange(L)) % N

            P=spectrum[idx_P]
            Q=spectrum[idx_Q]
            rho[i,d]=pcc_complex(P,Q)
    return rho

def _cyclic_window_sum(values: np.ndarray, window: int) -> np.ndarray:
    """sum of `window` consecutive cyclic values, for every start index."""
    values = np.asarray(values)
    n = values.size
    if not (1 <= window <= n):
        raise ValueError("window must be in [1, len(values)]")
    extended = np.concatenate((values, values[: window - 1]))
    prefix = np.empty(extended.size + 1, dtype=np.result_type(values.dtype, np.float64))
    prefix[0] = 0
    np.cumsum(extended, out=prefix[1:])
    return prefix[window:] - prefix[:-window]


def _correlation_axis(
    spectrum: np.ndarray,
    patch_shape: tuple[int, int],
    axis: int,
    eps: float,
) -> np.ndarray:
    """Correlate every non-overlapping 2D patch with all cyclic shifts on one axis.

    axis=0: vertical shifts, output shape (patch_rows, patch_cols, H).
    axis=1: horizontal shifts, output shape (patch_rows, patch_cols, W).

    The raw patch inner-products for all shifts are computed using FFTs.
    Patch sums and energies for mean-centering are computed with cyclic
    sliding sums. This is the fast 2D equivalent of applying pcc_complex to
    each patch pair independently.
    """
    F = np.asarray(spectrum, dtype=np.complex128)
    H, W = F.shape
    h, w = patch_shape
    npr, npc = H // h, W // w
    length = H if axis == 0 else W
    area = float(h * w)
    shifts = np.arange(length)
    rho = np.zeros((npr, npc, length), dtype=np.float64)

    if axis == 0:
        # For each patch-column block, prepare sums and energies for every
        # possible cyclic vertical starting row.
        sum_windows = []
        energy_windows = []
        for n in range(npc):
            c0, c1 = n * w, (n + 1) * w
            sum_rows = F[:, c0:c1].sum(axis=1)
            energy_rows = np.abs(F[:, c0:c1]) ** 2
            energy_rows = energy_rows.sum(axis=1)
            sum_windows.append(_cyclic_window_sum(sum_rows, h))
            energy_windows.append(_cyclic_window_sum(energy_rows, h))

        # Each anchor row block is handled as a batch over all image columns.
        for m in range(npr):
            r0 = m * h
            rotated = np.roll(F, shift=-r0, axis=0)
            padded = np.zeros_like(rotated)
            padded[:h, :] = rotated[:h, :]
            A = np.fft.fft(padded, axis=0)
            B = np.fft.fft(rotated, axis=0)
            # raw[d, c] = sum_p P[p,c] * conj(F[(r0+p+d) mod H,c])
            raw = np.conj(np.fft.ifft(np.conj(A) * B, axis=0))

            for n in range(npc):
                c0, c1 = n * w, (n + 1) * w
                raw_dot = raw[:, c0:c1].sum(axis=1)
                sw = sum_windows[n]
                ew = energy_windows[n]
                q_sum = sw[(r0 + shifts) % H]
                q_energy = ew[(r0 + shifts) % H]
                p_sum = sw[r0]
                p_energy = ew[r0]

                centered_dot = raw_dot - p_sum * np.conj(q_sum) / area
                p_var = max(float(np.real(p_energy - abs(p_sum) ** 2 / area)), 0.0)
                q_var = np.maximum(np.real(q_energy - np.abs(q_sum) ** 2 / area), 0.0)
                denom = np.sqrt(p_var * q_var)
                rho[m, n, :] = np.clip(np.abs(centered_dot) / (denom + eps), 0.0, 1.0)

    elif axis == 1:
        # For each patch-row block, prepare sums and energies for every
        # possible cyclic horizontal starting column.
        sum_windows = []
        energy_windows = []
        for m in range(npr):
            r0, r1 = m * h, (m + 1) * h
            sum_cols = F[r0:r1, :].sum(axis=0)
            energy_cols = (np.abs(F[r0:r1, :]) ** 2).sum(axis=0)
            sum_windows.append(_cyclic_window_sum(sum_cols, w))
            energy_windows.append(_cyclic_window_sum(energy_cols, w))

        for n in range(npc):
            c0 = n * w
            rotated = np.roll(F, shift=-c0, axis=1)
            padded = np.zeros_like(rotated)
            padded[:, :w] = rotated[:, :w]
            A = np.fft.fft(padded, axis=1)
            B = np.fft.fft(rotated, axis=1)
            # raw[r, d] = sum_q P[r,q] * conj(F[r,(c0+q+d) mod W])
            raw = np.conj(np.fft.ifft(np.conj(A) * B, axis=1))

            for m in range(npr):
                r0, r1 = m * h, (m + 1) * h
                raw_dot = raw[r0:r1, :].sum(axis=0)
                sw = sum_windows[m]
                ew = energy_windows[m]
                q_sum = sw[(c0 + shifts) % W]
                q_energy = ew[(c0 + shifts) % W]
                p_sum = sw[c0]
                p_energy = ew[c0]

                centered_dot = raw_dot - p_sum * np.conj(q_sum) / area
                p_var = max(float(np.real(p_energy - abs(p_sum) ** 2 / area)), 0.0)
                q_var = np.maximum(np.real(q_energy - np.abs(q_sum) ** 2 / area), 0.0)
                denom = np.sqrt(p_var * q_var)
                rho[m, n, :] = np.clip(np.abs(centered_dot) / (denom + eps), 0.0, 1.0)
    else:
        raise ValueError("axis must be 0 (vertical) or 1 (horizontal)")

    return rho



