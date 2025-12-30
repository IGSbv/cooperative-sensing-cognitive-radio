import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# IMPORT YOUR MODULES
from phase2_multiuser_data import generate_cooperative_data
from phase3_hmm_training import train_local_hmms, calculate_likelihoods
from phase4_bayesian_fusion import centralized_fusion, make_hard_decision
import phase2_multiuser_data # Import the module object to modify global variables

def run_simulation_sweep():
    # Define the range of SNRs to test
    snr_range = range(-20, 6, 2) # From -20dB to 4dB in steps of 2
    
    # Store results
    single_user_acc = []
    coop_acc = []
    
    print("--- Starting Performance Sweep (-20dB to 5dB) ---")
    
    for snr in snr_range:
        print(f"Testing SNR = {snr} dB...", end=" ")
        
        # 1. MODIFY GLOBAL SNR IN PHASE 2 MODULE
        # This is a hack to change the variable inside the other file dynamically
        phase2_multiuser_data.SNR_dB = snr
        
        # 2. RUN PIPELINE
        # Generate Data
        train_data, train_labels, test_data, test_labels = phase2_multiuser_data.generate_cooperative_data()
        
        # Train HMMs (Silent mode to reduce clutter)
        # We suppress prints by just running the function
        user_hmms = train_local_hmms(train_data)
        
        # Calculate Likelihoods
        local_llrs = calculate_likelihoods(user_hmms, test_data)
        
        # Fusion
        global_llr = centralized_fusion(local_llrs)
        
        # 3. CALCULATE ACCURACY
        # User 1 (Single Node)
        acc_u1 = accuracy_score(test_labels, make_hard_decision(local_llrs[:, 0]))
        # Cooperative (Fusion)
        acc_coop = accuracy_score(test_labels, make_hard_decision(global_llr))
        
        single_user_acc.append(acc_u1)
        coop_acc.append(acc_coop)
        
        print(f"Done. (Gain: {(acc_coop - acc_u1)*100:.1f}%)")
        
    return snr_range, single_user_acc, coop_acc

if __name__ == "__main__":
    # RUN THE SWEEP
    snr_axis, acc_single, acc_coop = run_simulation_sweep()
    
    # PLOT THE RESULTS
    plt.figure(figsize=(10, 6))
    
    # Plot Single User
    plt.plot(snr_axis, acc_single, 'r--o', label='Single User (Local HMM)', linewidth=2)
    
    # Plot Cooperative
    plt.plot(snr_axis, acc_coop, 'b-s', label='Cooperative Fusion (Proposed)', linewidth=2)
    
    # Formatting
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('Signal-to-Noise Ratio (dB)', fontsize=12)
    plt.ylabel('Detection Accuracy', fontsize=12)
    plt.title('Performance Comparison: Cooperative vs Single Node', fontsize=14)
    plt.legend(fontsize=12)
    plt.ylim(0.4, 1.05) # Accuracy is between 0.4 (random) and 1.0
    
    # Highlight the "Gain Zone"
    plt.fill_between(snr_axis, acc_single, acc_coop, color='green', alpha=0.1, label='Cooperative Gain')
    
    plt.show()
    print("Phase 5 Complete. You now have the final result graph for your report.")