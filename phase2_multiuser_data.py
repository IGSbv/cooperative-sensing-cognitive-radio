import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
NUM_USERS = 5            # Number of Cooperative Nodes
NUM_SAMPLES = 2000       # More samples for training
SNR_dB = 5             # Keep it challenging
TRAIN_SPLIT = 0.5        # 50% data for training HMM, 50% for testing

# --- REUSING PHASE 1 FUNCTIONS ---
# (Copy the functions generate_primary_user_signal and apply_fading_channel here 
#  OR import them if you know how. For safety, I'll redefine them briefly for context)

def generate_primary_user_signal(n_samples, prob_on=0.5):
    activities = np.random.choice([0, 1], size=n_samples, p=[1-prob_on, prob_on])
    bits = np.random.choice([-1, 1], size=n_samples)
    signals = activities * bits
    return activities, signals

def apply_fading_channel(signal, snr_db):
    n = len(signal)
    h = (np.random.normal(0, 1, n) + 1j * np.random.normal(0, 1, n)) / np.sqrt(2)
    snr_linear = 10**(snr_db / 10)
    noise_power = 1.0 / snr_linear
    noise = (np.random.normal(0, np.sqrt(noise_power/2), n) + 
             1j * np.random.normal(0, np.sqrt(noise_power/2), n))
    return h * signal + noise,h

# --- PHASE 2 LOGIC ---

# Replace the 'generate_cooperative_data' function in phase2_multiuser_data.py
# with this updated version that performs TIME AVERAGING.

def generate_cooperative_data():
    print(f"--- Generating Data for {NUM_USERS} Users with Time Averaging ---")
    
    # PARAMETERS
    SAMPLES_PER_DECISION = 10  # We average 10 raw samples for 1 HMM observation
    TOTAL_RAW_SAMPLES = NUM_SAMPLES * SAMPLES_PER_DECISION
    
    # 1. Generate Raw High-Speed Signal
    true_states_raw, tx_signal_raw = generate_primary_user_signal(TOTAL_RAW_SAMPLES)
    
    # Prepare arrays for the Averaged Data (What the HMM will see)
    averaged_energy = np.zeros((NUM_SAMPLES, NUM_USERS))
    averaged_states = np.zeros(NUM_SAMPLES)
    
    # 2. Process for each user
    for u in range(NUM_USERS):
        # Apply Channel
        rx_signal_raw, _ = apply_fading_channel(tx_signal_raw, SNR_dB)
        
        # Calculate Energy for blocks of 10 samples
        energy_raw = np.abs(rx_signal_raw)**2
        
        # Reshape into [NUM_SAMPLES, 10] and take the MEAN across axis 1
        # This is the "Integration" step that kills noise!
        averaged_energy[:, u] = np.mean(energy_raw.reshape(-1, SAMPLES_PER_DECISION), axis=1)
        
    # 3. Downsample the True States to match
    # If the PU was ON for the majority of the window, we call it ON.
    states_reshaped = true_states_raw.reshape(-1, SAMPLES_PER_DECISION)
    averaged_states = (np.mean(states_reshaped, axis=1) > 0.5).astype(int)
        
    # 4. Split Train/Test
    split_idx = int(NUM_SAMPLES * TRAIN_SPLIT)
    
    train_energy = averaged_energy[:split_idx, :]
    train_states = averaged_states[:split_idx]
    
    test_energy = averaged_energy[split_idx:, :]
    test_states = averaged_states[split_idx:]
    
    print(f"Data Generated. Shape: {train_energy.shape}. (Averaged {SAMPLES_PER_DECISION} raw samples per point)")
    return train_energy, train_states, test_energy, test_states

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels = generate_cooperative_data()
    
    # Visualization: Compare User 1 vs User 2
    plt.figure(figsize=(10, 5))
    plt.plot(train_data[:100, 0], label='User 1 Energy', alpha=0.7)
    plt.plot(train_data[:100, 1], label='User 2 Energy', alpha=0.7, color='red')
    plt.title("Spatial Diversity: Notice User 1 and User 2 see different energies!")
    plt.legend()
    plt.show()
    
    # SAVE THE DATA (Optional, but good practice)
    # np.savez('cooperative_data.npz', train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)