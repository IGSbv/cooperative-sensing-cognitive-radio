import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque

# IMPORT YOUR MODULES
from phase2_multiuser_data import generate_primary_user_signal, apply_fading_channel
from phase3_hmm_training import train_local_hmms
import phase2_multiuser_data

# --- CONFIGURATION ---
SNR_DB = -5                  # Challenging but visible
WINDOW_SIZE = 100            # How many points to show on screen (scrolling window)
UPDATE_INTERVAL_MS = 100     # Update speed (lower = faster)
SAMPLES_TO_AVG = 50          # Same averaging logic as your successful Phase 5

# GLOBAL BUFFERS (To hold data for plotting)
# Deque is a list that automatically pops old items when full
data_buffer = {
    'true_state': deque(np.zeros(WINDOW_SIZE), maxlen=WINDOW_SIZE),
    'user1_llr': deque(np.zeros(WINDOW_SIZE), maxlen=WINDOW_SIZE),
    'fusion_llr': deque(np.zeros(WINDOW_SIZE), maxlen=WINDOW_SIZE)
}

# GLOBAL OBJECTS
trained_hmms = []

def init_system():
    """
    1. Generates a small batch of 'Training Data' to pre-train the HMMs.
    2. Returns the trained models.
    """
    print("--- System Startup: Pre-training Models ---")
    # Temporarily generate 1000 samples just for training
    phase2_multiuser_data.SNR_dB = SNR_DB
    train_dat, _, _, _ = phase2_multiuser_data.generate_cooperative_data()
    
    # Train the brains once
    models = train_local_hmms(train_dat)
    print("--- System Ready: Starting Live Stream ---")
    return models

def get_live_data_chunk(frame):
    """
    Generates ONE time-step of data (Instantaneous)
    """
    # 1. Generate 1 "Block" of raw samples (e.g., 50 samples)
    # We force the PU to stay in one state for at least 20 frames to make it readable
    current_time = frame
    if (current_time // 20) % 2 == 0:
        true_state = 1 # ON
    else:
        true_state = 0 # OFF
        
    # Generate the raw signal for this block
    _, tx_signal = generate_primary_user_signal(SAMPLES_TO_AVG, prob_on=true_state)
    
    # 2. Pass through Channel for 5 users
    current_energy_avg = np.zeros((1, 5)) # [1 sample x 5 users]
    
    for u in range(5):
        rx, _ = apply_fading_channel(tx_signal, SNR_DB)
        energy_val = np.mean(np.abs(rx)**2)
        current_energy_avg[0, u] = energy_val
        
    return true_state, current_energy_avg

def update_plot(frame):
    """
    The Loop function called by Matplotlib every 100ms
    """
    # 1. Get Live Data
    true_state, live_energy = get_live_data_chunk(frame)
    
    # 2. HMM Inference (Calculate LLRs)
    local_llrs = np.zeros((1, 5))
    
    for u in range(5):
        model = trained_hmms[u]
        data_point = live_energy[:, u].reshape(-1, 1)
        
        # Calculate Probabilities
        probs = model.predict_proba(data_point)
        
        # Identify ON state (Higher Mean)
        if model.means_[1][0] > model.means_[0][0]:
            idx_on = 1; idx_off = 0
        else:
            idx_on = 0; idx_off = 1
            
        p_on = probs[:, idx_on] + 1e-10
        p_off = probs[:, idx_off] + 1e-10
        local_llrs[:, u] = np.log(p_on / p_off)
        
    # 3. Fusion
    global_llr = np.sum(local_llrs)
    
    # 4. Update Buffers
    data_buffer['true_state'].append(true_state)
    data_buffer['user1_llr'].append(local_llrs[0, 0]) # User 1 opinion
    data_buffer['fusion_llr'].append(global_llr)      # Cooperative opinion
    
    # 5. Update Lines on Plot
    line_true.set_ydata(data_buffer['true_state'])
    line_user1.set_ydata(data_buffer['user1_llr'])
    line_fusion.set_ydata(data_buffer['fusion_llr'])
    
    # Dynamic Scaling (Optional: keep y-axis fitting the data)
    ax2.set_ylim(min(data_buffer['user1_llr'])-2, max(data_buffer['user1_llr'])+2)
    ax3.set_ylim(min(data_buffer['fusion_llr'])-2, max(data_buffer['fusion_llr'])+2)
    
    return line_true, line_user1, line_fusion

# --- MAIN REAL-TIME PLOT SETUP ---
if __name__ == "__main__":
    # 1. Train first
    trained_hmms = init_system()
    
    # 2. Setup Plot Layout
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8))
    
    # Plot 1: True State (Green)
    ax1.set_title(f"REAL-TIME SIMULATION (SNR = {SNR_DB} dB)")
    ax1.set_ylabel("True PU")
    ax1.set_ylim(-0.2, 1.2)
    line_true, = ax1.plot(np.zeros(WINDOW_SIZE), color='green', linewidth=2)
    
    # Plot 2: Single User (Purple)
    ax2.set_ylabel("User 1 LLR")
    line_user1, = ax2.plot(np.zeros(WINDOW_SIZE), color='purple', alpha=0.6)
    ax2.axhline(0, color='black', linestyle='--')
    
    # Plot 3: Fusion Center (Blue)
    ax3.set_ylabel("Fusion LLR")
    ax3.set_xlabel("Time (Scrolling Window)")
    line_fusion, = ax3.plot(np.zeros(WINDOW_SIZE), color='blue', linewidth=2)
    ax3.axhline(0, color='black', linestyle='--')
    
    # 3. Start Animation
    # interval=100 means 100ms (10 frames per second)
    ani = FuncAnimation(fig, update_plot, interval=UPDATE_INTERVAL_MS, blit=False)
    
    plt.tight_layout()
    plt.show()