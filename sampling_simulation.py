"""
Sampling Theorem and Aliasing Simulation
Course: Signals and Systems

Requirements:
    pip install numpy matplotlib

Run:
    python sampling_simulation.py

This program:
1. Generates a 5 Hz sine wave.
2. Samples it at 50 Hz, 12 Hz and 7 Hz.
3. Demonstrates the Nyquist criterion and aliasing.
4. Performs ideal sinc interpolation for visualization.
5. Saves four figures in a folder named 'figures'.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

AMPLITUDE = 1.0
SIGNAL_FREQUENCY = 5.0       # Hz
DURATION = 1.0               # seconds
SAMPLE_RATES = [50.0, 12.0, 7.0]
OUTPUT_DIR = "figures"

def continuous_signal(time):
    """Return x(t) = A sin(2*pi*f*t)."""
    return AMPLITUDE * np.sin(2 * np.pi * SIGNAL_FREQUENCY * time)

def sample_signal(sampling_frequency):
    """Generate discrete samples of the continuous sine wave."""
    sample_time = np.arange(0, DURATION, 1 / sampling_frequency)
    samples = continuous_signal(sample_time)
    return sample_time, samples

def ideal_sinc_reconstruction(sample_time, samples, sampling_frequency, time):
    """Reconstruct a band-limited signal using ideal sinc interpolation."""
    return np.sum(
        samples[:, None] *
        np.sinc(sampling_frequency * (time[None, :] - sample_time[:, None])),
        axis=0
    )

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    time = np.linspace(0, DURATION, 4000, endpoint=False)
    original = continuous_signal(time)
    nyquist_rate = 2 * SIGNAL_FREQUENCY

    print("SAMPLING THEOREM AND ALIASING")
    print(f"Signal frequency: {SIGNAL_FREQUENCY:.2f} Hz")
    print(f"Nyquist rate: {nyquist_rate:.2f} Hz")
    for fs in SAMPLE_RATES:
        status = "Satisfies Nyquist criterion" if fs > nyquist_rate else "Aliasing may occur"
        print(f"Sampling frequency = {fs:.2f} Hz -> {status}")

    # Figure 1: original signal
    plt.figure(figsize=(9, 4.5))
    plt.plot(time, original, linewidth=2, label="Original signal")
    plt.title("Original 5 Hz Sine Wave")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_original_signal.png"), dpi=180)
    plt.close()

    # Figure 2: sampling comparison
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    labels = ["Above Nyquist rate", "Near Nyquist rate", "Below Nyquist rate — aliasing"]
    for ax, fs, label in zip(axes, SAMPLE_RATES, labels):
        sample_time, samples = sample_signal(fs)
        ax.plot(time, original, color="black", alpha=0.55, label="Original signal")
        ax.stem(sample_time, samples, linefmt="C1-", markerfmt="C1o",
                basefmt=" ", label=f"Samples at {fs:g} Hz")
        ax.set_title(f"{label} ({fs:g} Hz)")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (seconds)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "02_sampling_comparison.png"), dpi=180)
    plt.close(fig)

    # Figure 3: sinc reconstruction comparison
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for ax, fs in zip(axes, [50.0, 7.0]):
        sample_time, samples = sample_signal(fs)
        reconstructed = ideal_sinc_reconstruction(sample_time, samples, fs, time)
        ax.plot(time, original, linewidth=2, label="Original signal")
        ax.plot(time, reconstructed, "--", linewidth=1.7, label="Sinc reconstruction")
        ax.stem(sample_time, samples, linefmt="C2-", markerfmt="C2o",
                basefmt=" ", label="Samples")
        ax.set_title(f"Sinc reconstruction at {fs:g} Hz")
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right")
    axes[-1].set_xlabel("Time (seconds)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "03_reconstruction.png"), dpi=180)
    plt.close(fig)

    # Figure 4: alias frequency
    fs = 7.0
    alias_frequency = abs(SIGNAL_FREQUENCY - round(SIGNAL_FREQUENCY / fs) * fs)
    plt.figure(figsize=(9, 4.5))
    plt.plot(time, original, linewidth=2, label=f"Original: {SIGNAL_FREQUENCY:g} Hz")
    plt.plot(time, np.sin(2*np.pi*alias_frequency*time), "--", linewidth=2,
             label=f"Apparent alias: {alias_frequency:g} Hz")
    plt.title("Aliasing: Frequency Misinterpretation")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_alias_frequency.png"), dpi=180)
    plt.close()

    print(f"\nFour figures saved in: {OUTPUT_DIR}/")
    print("Simulation completed successfully.")

if __name__ == "__main__":
    main()
