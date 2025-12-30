import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

# IMPORT DATA FROM PHASE 2
# This automatically runs the phase2 script to generate fresh data
from phase2_multiuser_data import generate_cooperative_data

def train_local_hmms(train_energy):
    """
    Trains an HMM for each user independently.
    """
    print("--- Training Local HMMs for Each User ---")
    num_users = train_energy.shape[1]
    hmms = []

    for u in range(num_users):
        # 1. Initialize HMM (GaussianHMM because Energy is continuous)
        # n_components=2 corresponds to [PU_OFF, PU_ON]
        # n_iter=100 allows enough time for the model to converge
        model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=100)
        
        # 2. Reshape data for hmmlearn (Needs a column vector)
        user_data = train_energy[:, u].reshape(-1, 1)
        
        # 3. Train the model (Unsupervised Learning)
        # The HMM figures out "High Energy State" vs "Low Energy State" on its own
        model.fit(user_data)
        
        hmms.append(model)
        
    print(f"Successfully trained {len(hmms)} HMM models.")
    return hmms

def calculate_likelihoods(hmms, test_energy):
    """
    Returns the Log-Likelihood Ratio (LLR) for each user.
    LLR = log( P(Observation | ON) / P(Observation | OFF) )
    """
    num_samples = test_energy.shape[0]
    num_users = test_energy.shape[1]
    
    # Store LLRs: [Samples x Users]
    llrs = np.zeros((num_samples, num_users))
    
    for u in range(num_users):
        model = hmms[u]
        data = test_energy[:, u].reshape(-1, 1)
        
        # 1. Identify which state is 'ON'
        # HMM assigns State 0 and State 1 randomly.
        # We assume the state with the HIGHER Mean Energy is the 'ON' state.
        if model.means_[1][0] > model.means_[0][0]:
            idx_on = 1
            idx_off = 0
        else:
            idx_on = 0
            idx_off = 1
            
        # 2. Get Probabilities for each state
        # predict_proba returns [P(State0), P(State1)] for every sample
        probs = model.predict_proba(data)
        
        # 3. Calculate Log Likelihood Ratio
        # Add tiny epsilon to avoid log(0) crashing the code
        epsilon = 1e-10
        p_on = probs[:, idx_on] + epsilon
        p_off = probs[:, idx_off] + epsilon
        
        # LLR formula
        llrs[:, u] = np.log(p_on / p_off)
        
    return llrs

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # 1. Get Data from Phase 2
    train_data, train_labels, test_data, test_labels = generate_cooperative_data()
    
    # 2. Train the HMMs
    user_hmms = train_local_hmms(train_data)
    
    # 3. Calculate Soft Decisions (LLR)
    print("--- Calculating Likelihood Ratios ---")
    llrs = calculate_likelihoods(user_hmms, test_data)
    
    # 4. Visualization: LLR vs True State
    # We plot the LLR for User 1. 
    # Positive spikes should align with Green Blocks (True ON state).
    
    plt.figure(figsize=(12, 6))
    
    # Subplot 1: True Activity
    plt.subplot(2, 1, 1)
    plt.step(range(100), test_labels[:100], where='post', color='green', label='True PU Activity')
    plt.ylabel("State")
    plt.title("True Primary User Activity (First 100 Test Samples)")
    plt.legend()
    
    # Subplot 2: User 1's Opinion (LLR)
    plt.subplot(2, 1, 2)
    plt.plot(llrs[:100, 0], color='purple', label='User 1 LLR')
    plt.axhline(0, color='black', linestyle='--', alpha=0.5) # Zero line
    plt.ylabel("Log Likelihood Ratio")
    plt.title("User 1 Soft Decision (HMM Output)")
    plt.legend()
    
    plt.tight_layout()
    plt.show()
    
    print("Phase 3 Complete: HMMs Trained and LLRs Calculated.")