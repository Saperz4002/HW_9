"""HW9: 1D Yee FDTD with conductivity, Debye dispersion, and a finite slab.

Run: python hw9_fdtd.py [all|lossy|dielectric|debye|slab|raw|abc]
Dependencies: NumPy and Matplotlib (Pillow for optional GIFs).
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

C0 = 299792458.0
EPS0 = 8.8541878128e-12
# Slide 15 and slide 21 use the same material.
EPS_REAL, SIGMA, CHI1, TAU = 2.0, 0.01, 2.0, 1e-9
SLAB_THICKNESS = 0.5
LECTURE_CASES = ("lossy", "dielectric", "debye")
CASES = (*LECTURE_CASES, "slab", "raw", "abc")


def run_yee(ddx, courant, pulse, ksource, epsilon, sigma, chi1, tau,
            *, probes=(), save_fields=False, boundary_delays=(2, 2),
            hard_source=False):
    """Advance slide variables dx, ex, ix, sx, hy on a staggered grid.

    Row n of Ex is E at n*dt. Row n of Hy for n >= 1 is H at
    (n + 1/2)*dt. Both row-zero arrays are initialized to zero.
    A boundary_delays value of None holds the endpoints at zero.
    """
    ke = epsilon.size
    dt = courant * ddx / C0
    gb = sigma * dt / EPS0
    gc = chi1 * dt / tau
    delexp = np.exp(-dt / tau)
    ga = 1.0 / (epsilon + gb + gc)

    dx = np.zeros(ke)
    ex = np.zeros(ke)
    ix = np.zeros(ke)
    sx = np.zeros(ke)
    hy = np.zeros(ke - 1)  # Only physical staggered H nodes.
    samples = pulse.size + 1
    probe_indices = np.array(probes, dtype=int)
    trace = np.zeros((samples, len(probes)))
    E = np.zeros((samples, ke)) if save_fields else None
    H = np.zeros((samples, ke - 1)) if save_fields else None
    if boundary_delays is not None:
        left, right = boundary_delays
        low_history = np.zeros(left)
        high_history = np.zeros(right)

    interior = slice(1, -1)
    for n, source_value in enumerate(pulse, start=1):
        dx[interior] += courant * (hy[:-1] - hy[1:])
        if not hard_source:
            dx[ksource] += source_value

        ex[interior] = ga[interior] * (
            dx[interior] - ix[interior] - delexp * sx[interior]
        )
        if hard_source:
            # The HW8 hard source is in vacuum, where E_x = D_x.
            ex[ksource] = dx[ksource] = source_value
        ix[interior] += gb[interior] * ex[interior]
        sx[interior] = delexp * sx[interior] + gc[interior] * ex[interior]

        if boundary_delays is not None:
            low_slot, high_slot = n % left, n % right
            ex[0], ex[-1] = low_history[low_slot], high_history[high_slot]
            low_history[low_slot], high_history[high_slot] = ex[1], ex[-2]

        hy += courant * (ex[:-1] - ex[1:])
        if probes:
            trace[n] = ex[probe_indices]
        if save_fields:
            E[n], H[n] = ex, hy

    return E, H, trace


def fourier_samples(fields, time, frequencies):
    """Evaluate the discrete transform with the exp(-i omega t) convention."""
    values = np.empty((len(frequencies), fields.shape[1]), dtype=complex)
    # Frequency blocks limit the size of the temporary phase matrix.
    for start in range(0, len(frequencies), 32):
        block = slice(start, start + 32)
        phase = np.exp(-2j * np.pi * frequencies[block, None] * time[None, :])
        values[block] = phase @ fields
    return values


def lecture_example(mode, folder, gif):
    ke, kstart, ddx, courant = 200, 100, 0.01, 0.5
    dt = courant * ddx / C0
    epsilon, sigma, chi1 = np.ones(ke), np.zeros(ke), np.zeros(ke)
    hard_source = mode in ("raw", "abc")
    ksource, t0, spread = (25, 40, 10) if hard_source else (5, 50, 10)
    boundary_delays = None if mode == "raw" else (2, 2)
    frequencies = np.empty(0)
    if mode in ("lossy", "dielectric"):
        epsilon[kstart:] = 4.0
        boundary_delays = (2, 4)
    if mode == "lossy":
        sigma[kstart:] = 0.04
        snapshots, nsteps = (500,), 500
    elif mode == "dielectric":
        frequencies = np.array([500e6])
        snapshots, nsteps = (200, 400), 2400
    elif mode == "debye":
        epsilon[kstart:], sigma[kstart:], chi1[kstart:] = EPS_REAL, SIGMA, CHI1
        boundary_delays = (2, 4)
        frequencies = np.array([50e6, 200e6, 500e6])
        snapshots, nsteps = (250, 1000), 3600
    else:
        snapshots, nsteps = (200, 400), 1400

    steps = np.arange(1, nsteps + 1)
    if mode == "lossy":
        pulse = np.sin(2 * np.pi * 700e6 * steps * dt)
    else:
        pulse = np.exp(-0.5 * ((steps - t0) / spread) ** 2)

    E, H, _ = run_yee(
        ddx, courant, pulse, ksource, epsilon, sigma, chi1, TAU,
        save_fields=True, boundary_delays=boundary_delays,
        hard_source=hard_source,
    )
    k = np.arange(ke)
    time = np.arange(nsteps + 1) * dt
    output = dict(
        E_tilde=E, H=H, z_E=k * ddx, z_H=(k[:-1] + 0.5) * ddx,
        t_E=time, t_H=time + dt / 2, eps_r=epsilon, sigma=sigma,
        chi1=chi1, tau=TAU, dx=ddx, dt=dt, S=courant, c0=C0, eps0=EPS0,
        mode=mode, source_index=ksource,
        interface_index=-1 if hard_source else kstart,
        source_frequency=700e6 if mode == "lossy" else np.nan,
        gaussian_center_step=t0, gaussian_spread_steps=spread,
        source_type="hard" if hard_source else "soft",
        boundary_type="reflecting" if mode == "raw" else "approximate_delay",
    )
    if frequencies.size:
        # The vacuum reference supplies the incident spectrum at the source.
        _, _, incident = run_yee(
            ddx, courant, pulse, ksource, np.ones(ke), np.zeros(ke),
            np.zeros(ke), TAU, probes=(ksource,),
        )
        spectrum = fourier_samples(E, time, frequencies)
        reference = fourier_samples(incident, time, frequencies)
        normalized = spectrum / reference
        output.update(frequency_hz=frequencies, amplitude=np.abs(normalized),
                      phase_rad=np.angle(normalized))
    np.savez_compressed(folder / f"yee_{mode}_fields.npz", **output)

    def decorate(ax):
        if not hard_source:
            ax.axvspan(kstart, ke - 1, color="#008c95", alpha=0.12)
        ax.axvline(ksource, color="0.55", ls=":", linewidth=1)
        ax.set(xlim=(0, ke - 1), xlabel="Grid index $k$")
        ax.grid(alpha=0.22)

    count = len(snapshots) + 2 * bool(frequencies.size)
    fig, axes = plt.subplots(count, 1, figsize=(8, 2.35 * count), layout="constrained")
    for ax, step in zip(np.atleast_1d(axes), snapshots):
        ax.plot(k, E[step], color="#213b55")
        ax.set(ylabel=r"$\widetilde E_x$", title=f"{mode.capitalize()} · step {step}")
        decorate(ax)
    if frequencies.size:
        for index, f in enumerate(frequencies):
            label = f"{f / 1e6:g} MHz"
            axes[-2].plot(k, np.abs(normalized[index]), label=label)
            axes[-1].plot(k, np.angle(normalized[index]), label=label)
        axes[-2].set(ylabel="Relative amplitude")
        axes[-1].set(ylabel="Relative phase (rad)")
        axes[-2].legend()
        decorate(axes[-2])
        decorate(axes[-1])
    fig.savefig(folder / f"yee_{mode}.pdf")
    plt.close(fig)

    if gif:
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        decorate(ax)
        limit = max(1e-9, 1.1 * np.max(np.abs(E)))
        ax.set(ylim=(-limit, limit), ylabel=r"$\widetilde E_x$")
        line, = ax.plot(k, E[0], color="#213b55")

        def update(n):
            line.set_ydata(E[n])
            ax.set_title(f"{mode.capitalize()} · step {n} · {time[n]*1e9:.2f} ns")
            return (line,)

        stride = max(1, nsteps // 150)
        frames = [*range(0, nsteps, stride), nsteps]
        movie = FuncAnimation(fig, update, frames=frames)
        movie.save(folder / f"yee_{mode}.gif", writer=PillowWriter(fps=25))
        plt.close(fig)


def analytical_slab(frequencies):
    """Vacuum–slab–vacuum Fresnel solution for the slides' +iωt convention."""
    omega = 2 * np.pi * frequencies
    epsilon = EPS_REAL + SIGMA / (1j * omega * EPS0) + CHI1 / (1 + 1j * omega * TAU)
    index = np.sqrt(epsilon)
    r01 = (1 - index) / (1 + index)
    t01 = 2 / (1 + index)
    t10 = 2 * index / (1 + index)
    # A forward wave has exp(-i kz). Im(index) < 0 for this passive medium.
    p = np.exp(-1j * omega * index * SLAB_THICKNESS / C0)
    denominator = 1 - r01**2 * p**2
    r = r01 * (1 - p**2) / denominator
    t = t01 * t10 * p / denominator
    R, T = np.abs(r)**2, np.abs(t)**2
    return np.column_stack((R, T, 1 - R - T))


def slab_setup(ddx, duration):
    """Build the shared slab geometry and Gaussian source at Courant 1."""
    dt = ddx / C0
    z = np.arange(round(2.0 / ddx) + 1) * ddx
    ke = z.size
    epsilon, sigma, chi1 = np.ones(ke), np.zeros(ke), np.zeros(ke)
    slab_left = (2.0 - SLAB_THICKNESS) / 2
    slab_right = slab_left + SLAB_THICKNESS
    in_slab = slice(round(slab_left / ddx), round(slab_right / ddx))
    epsilon[in_slab], sigma[in_slab], chi1[in_slab] = EPS_REAL, SIGMA, CHI1

    time = np.arange(round(duration / dt) + 1) * dt
    # Short enough to excite 2 GHz and centered well after t = 0.
    pulse = np.exp(-0.5 * ((time[1:] - 0.5e-9) / 75e-12)**2)
    ksource = round(0.25 / ddx)
    return z, time, pulse, ksource, epsilon, sigma, chi1


def slab_spectra(ddx):
    # At Courant 1 a one-step neighbor copy absorbs an outgoing wave
    # exactly on the discrete vacuum grid at either end.
    courant = 1.0
    z, time, pulse, ksource, epsilon, sigma, chi1 = slab_setup(ddx, 100e-9)
    ke = z.size
    probes = (round(0.5 / ddx), round(1.75 / ddx))
    _, _, vacuum = run_yee(
        ddx, courant, pulse, ksource, np.ones(ke), np.zeros(ke),
        np.zeros(ke), TAU, probes=probes, boundary_delays=(1, 1),
    )
    _, _, slab = run_yee(
        ddx, courant, pulse, ksource, epsilon, sigma, chi1, TAU,
        probes=probes, boundary_delays=(1, 1),
    )
    frequencies = np.arange(1, 2001) * 1e6
    traces = np.column_stack((vacuum, slab[:, 0] - vacuum[:, 0], slab[:, 1]))
    spectra = fourier_samples(traces, time, frequencies)
    R = np.abs(spectra[:, 2] / spectra[:, 0])**2
    T = np.abs(spectra[:, 3] / spectra[:, 1])**2
    tail = np.max(np.abs(slab[time >= 0.9 * time[-1]])) / np.max(np.abs(vacuum))
    return frequencies, np.column_stack((R, T, 1 - R - T)), tail


def animate_slab(folder, ddx):
    """Save the first 13 ns of the same excitation used for the spectra."""
    z, time, pulse, ksource, epsilon, sigma, chi1 = slab_setup(ddx, 13e-9)
    E, _, _ = run_yee(
        ddx, 1.0, pulse, ksource, epsilon, sigma, chi1, TAU,
        save_fields=True, boundary_delays=(1, 1),
    )
    fig, ax = plt.subplots(figsize=(8, 3.4), layout="constrained")
    slab_left = (z[-1] - SLAB_THICKNESS) / 2
    ax.axvspan(slab_left, slab_left + SLAB_THICKNESS, color="#008c95",
               alpha=0.14, label="Lossy Debye slab")
    ax.axvline(z[ksource], color="0.5", linestyle=":", label="Source")
    ax.axhline(0, color="0.65", linewidth=0.7)
    limit = 1.2 * np.max(np.abs(E))
    ax.set(xlim=(z[0], z[-1]), ylim=(-limit, limit), xlabel="Position (m)",
           ylabel=r"$\widetilde{E}_x$", title="Gaussian pulse through a 0.5 m Debye slab")
    ax.legend(loc="upper right", frameon=False)
    line, = ax.plot(z, E[0], color="#213b55", linewidth=1.8)
    clock = ax.text(0.025, 0.91, "", transform=ax.transAxes, color="#213b55")

    def update(n):
        line.set_ydata(E[n])
        clock.set_text(f"{time[n] * 1e9:05.2f} ns")
        return line, clock

    frames = np.linspace(0, len(time) - 1, 160, dtype=int)
    movie = FuncAnimation(fig, update, frames=frames)
    movie.save(folder / "hw9_slab.gif", writer=PillowWriter(fps=24))
    plt.close(fig)


def plot_slab(folder, gif=False):
    ddx = 0.00125
    frequency, fdtd, tail = slab_spectra(ddx)
    exact = analytical_slab(frequency)
    error = np.max(np.abs(fdtd - exact), axis=0)
    np.savetxt(
        folder / "slab_spectra.csv", np.column_stack((frequency / 1e6, fdtd, exact)),
        delimiter=",", comments="",
        header="frequency_MHz,R_fdtd,T_fdtd,A_fdtd,R_analytical,T_analytical,A_analytical",
    )
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(8, 7), layout="constrained")
    for index, (ax, label) in enumerate(zip(axes, ("Reflection", "Transmission", "Absorption"))):
        ax.plot(frequency / 1e6, exact[:, index], color="#213b55", label="Analytical")
        ax.plot(frequency / 1e6, fdtd[:, index], "--", color="#008c95", label="FDTD")
        ax.set(ylabel=label, ylim=(-0.02, 1.02))
        ax.grid(alpha=0.22)
    axes[0].set_title("Vacuum–Debye slab–vacuum · 0.5 m slab")
    axes[0].legend()
    axes[-1].set(xlabel="Frequency (MHz)", xlim=(1, 2000))
    fig.savefig(folder / "slab_spectra.pdf")
    plt.close(fig)
    print(f"Slab maximum |FDTD - analytical|: R={error[0]:.4g}, "
          f"T={error[1]:.4g}, A={error[2]:.4g}")
    print(f"Slab probe tail relative to peak incident field: {tail:.2e}")
    if gif:
        animate_slab(folder, ddx)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", default="all", choices=("all", *CASES))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--gif", action="store_true", help="Animate the selected field examples.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    selected = (*LECTURE_CASES, "slab") if args.mode == "all" else (args.mode,)
    for mode in selected:
        if mode == "slab":
            plot_slab(args.output, args.gif)
        else:
            lecture_example(mode, args.output, args.gif)
    print(f"Saved outputs to {args.output.resolve()}")


if __name__ == "__main__":
    main()
