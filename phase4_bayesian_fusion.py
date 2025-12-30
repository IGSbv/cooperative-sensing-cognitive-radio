import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

# IMPORT PREVIOUS LOGIC
# We need to recreate the training/testing pipeline here to keep it clean
from phase2_multiuser_data import generate_cooperative_data
from phase3_hmm_training import train_local_hmms, calculate_likelihoods

def centralized_fusion(llrs_all_users):
    """
    Step 4 Core Logic: Bayesian Decision Fusion.
    In the Log-Domain, 'Product of Probabilities' becomes 'Sum of Logs'.
    
    Formula: Global_LLR = Sum(Local_LLR_user_i)
    """
    # Sum across the rows (axis 1) -> Summing opinions of all 5 users for each time slot
    global_llr = np.sum(llrs_all_users, axis=1)
    return global_llr

def make_hard_decision(llr_values, threshold=0):
    """
    Converts Soft LLR -> Hard Binary Decision (1 or 0)
    """
    decisions = np.zeros_like(llr_values)
    decisions[llr_values > threshold] = 1
    return decisions

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. GENERATE ENVIRONMENT
    train_data, train_labels, test_data, test_labels = generate_cooperative_data()
    
    # 2. LOCAL SENSING (HMM Training)
    user_hmms = train_local_hmms(train_data)
    
    # 3. LOCAL DECISIONS (Get LLRs)
    # This gives us a matrix [1000 samples x 5 users]
    local_llrs = calculate_likelihoods(user_hmms, test_data)
    
    # 4. CENTRALIZED FUSION
    global_llr = centralized_fusion(local_llrs)
    
    # 5. PERFORMANCE COMPARISON
    # Let's compare "User 1 Alone" vs "Cooperative Fusion"
    
    # Make Hard Decisions
    user1_decision = make_hard_decision(local_llrs[:, 0]) # User 1 only
    fusion_decision = make_hard_decision(global_llr)      # Cooperative
    
    # Calculate Accuracy
    acc_user1 = accuracy_score(test_labels, user1_decision)
    acc_fusion = accuracy_score(test_labels, fusion_decision)
    
    print("\n" + "="*40)
    print(f"RESULTS (SNR = -10 dB, Users = 5)")
    print("="*40)
    print(f"Single User Accuracy: {acc_user1 * 100:.2f}%")
    print(f"Cooperative Accuracy: {acc_fusion * 100:.2f}%")
    print(f"Improvement Gain:     {(acc_fusion - acc_user1) * 100:.2f}%")
    print("="*40 + "\n")
    
    # 6. VISUALIZATION
    plt.figure(figsize=(12, 8))
    
    # Plot 1: True State
    plt.subplot(3, 1, 1)
    plt.step(range(100), test_labels[:100], where='post', color='green', label='True PU')
    plt.ylabel("True State")
    plt.title("Truth (Primary User Activity)")
    plt.legend(loc='upper right')
    
    # Plot 2: Single User Mistake
    plt.subplot(3, 1, 2)
    plt.plot(local_llrs[:100, 0], color='purple', alpha=0.6, label='User 1 LLR')
    plt.axhline(0, color='black', linestyle='--')
    plt.fill_between(range(100), local_llrs[:100, 0], 0, where=(local_llrs[:100, 0] < 0) & (test_labels[:100]==1), color='red', alpha=0.3, label='Missed Detection')
    plt.ylabel("User 1 LLR")
    plt.title("Single User View (Red zones = Missed Detection)")
    plt.legend(loc='upper right')

    # Plot 3: Fusion Success
    plt.subplot(3, 1, 3)
    plt.plot(global_llr[:100], color='blue', linewidth=2, label='Global Fusion LLR')
    plt.axhline(0, color='black', linestyle='--')
    plt.ylabel("Global LLR")
    plt.title(f"Cooperative Fusion View (Accuracy: {acc_fusion*100:.1f}%)")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    plt.show()