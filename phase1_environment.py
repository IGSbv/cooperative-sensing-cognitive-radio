import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION PARAMETERS ---
NUM_SAMPLES = 1000       # Total time slots to simulate
PU_ON_PROB = 0.5         # Probability that Primary User is Active
SNR_dB = -10             # Signal-to-Noise Ratio (Critical for Cognitive Radio: -10dB is noisy!)
DOPPLER_SHIFT = 10       # Simulates how fast the channel changes (Fading speed)

def generate_primary_user_signal(n_samples, prob_on):
    """
    Step 1: Generate the 'True' Activity of the Primary User (PU).
    Returns:
        activities: Binary array (1 = ON, 0 = OFF)
        signals: The actual transmitted signal (Phase Shift Keying)
    """
    # 1. Generate Random States (ON or OFF)
    activities = np.random.choice([0, 1], size=n_samples, p=[1-prob_on, prob_on])
    
    # 2. Generate Complex Signal (BPSK Modulation)
    # If ON: Send +1 or -1 (random data). If OFF: Send 0.
    bits = np.random.choice([-1, 1], size=n_samples)
    signals = activities * bits  # If activity is 0, signal becomes 0
    
    return activities, signals

def apply_fading_channel(signal, snr_db):
    """
    Step 2: Simulate the Wireless Channel (Rayleigh Fading + Noise).
    Formula: y = h * x + n
    """
    n = len(signal)
    
    # 1. Generate Rayleigh Fading Coefficients (h)
    # A complex Gaussian variable (Real + Imaginary parts are random) creates Rayleigh magnitude
    h_real = np.random.normal(0, 1, n)
    h_imag = np.random.normal(0, 1, n)
    h = (h_real + 1j * h_imag) / np.sqrt(2) # Normalize power to 1
    
    # 2. Generate White Gaussian Noise (n)
    # Calculate Noise Power based on SNR
    # SNR_linear = Signal_Power / Noise_Power
    # Since Signal Power ~ 1 (due to BPSK and normalized h), Noise Power = 1 / SNR_linear
    snr_linear = 10**(snr_db / 10)
    noise_power = 1.0 / snr_linear
    
    noise_real = np.random.normal(0, np.sqrt(noise_power/2), n)
    noise_imag = np.random.normal(0, np.sqrt(noise_power/2), n)
    noise = noise_real + 1j * noise_imag
    
    # 3. Received Signal (y)
    # We multiply signal by channel (h) and add noise
    received_signal = h * signal + noise
    
    return received_signal, h

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Run PU Generator
    true_states, tx_signal = generate_primary_user_signal(NUM_SAMPLES, PU_ON_PROB)
    
    # 2. Pass through Channel
    rx_signal, channel_h = apply_fading_channel(tx_signal, SNR_dB)
    
    # 3. Visualization (Proof it works)
    plt.figure(figsize=(12, 6))
    
    # Plot 1: True PU Activity
    plt.subplot(3, 1, 1)
    plt.step(range(100), true_states[:100], where='post', color='green')
    plt.title("Step 1: True Primary User Activity (First 100 samples)")
    plt.ylabel("State (1=ON)")
    plt.ylim(-0.2, 1.2)
    
    # Plot 2: Fading Channel Magnitude
    plt.subplot(3, 1, 2)
    plt.plot(np.abs(channel_h[:100]), color='orange')
    plt.title("Step 2: Channel Fading Magnitude (|h|)")
    plt.ylabel("Gain")
    
    # Plot 3: Received Energy (What the Cognitive Radio 'Sees')
    energy = np.abs(rx_signal)**2
    plt.subplot(3, 1, 3)
    plt.plot(energy[:100], color='blue')
    plt.title(f"Step 3: Received Signal Energy (SNR = {SNR_dB} dB)")
    plt.xlabel("Time Sample")
    plt.ylabel("Energy")
    
    plt.tight_layout()
    plt.show()
    
    print("Phase 1 Complete: Environment Generated.")