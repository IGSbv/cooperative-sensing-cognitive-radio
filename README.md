# **Cooperative Spectrum Sensing in Cognitive Radio Networks**

### **Using Centralized Bayesian Decision Fusion and Distributed Hidden Markov Models**

## **📌 Project Overview**

This project addresses the critical challenge of **Spectrum Scarcity** in next-generation wireless networks (5G/6G). It implements a **Cognitive Radio (CR)** system that allows secondary users to opportunistically access licensed spectrum.

Standard sensing techniques (Energy Detection) fail under **Deep Fading** and low **SNR (-10 dB)** conditions. This project solves that problem by implementing a **Cooperative Sensing Framework** where multiple users fuse their local probabilistic decisions (HMMs) to make a robust global decision.

## **🚀 Key Features**

* **Physics-Compliant Simulation:** Implements Rayleigh Fading, Shadowing, and AWGN channels.  
* **Machine Learning Integration:** Uses **Hidden Markov Models (HMM)** for local temporal sensing (soft decision).  
* **Bayesian Data Fusion:** Centralized fusion center combines Log-Likelihood Ratios (LLR) from 5+ users.  
* **Robustness:** Achieves **\~20% accuracy gain** over single-node sensing at \-2 dB SNR.  
* **Real-Time Demo:** Includes a live-streaming dashboard visualizing detection in real-time.

## **📊 Visual Results & Methodology**

### **1\. The Environment (Physics Layer)**

We simulate a Primary User (PU) transmitting BPSK signals through a **Rayleigh Fading Channel**.

* **Green:** True PU Activity (Ground Truth).  
* **Orange:** Channel Fading Magnitude (Notice the deep fades near 0).  
* **Blue:** What the sensor actually sees (Noisy & Faded).

*(Run phase1\_environment.py to generate)*

### **2\. Spatial Diversity**

Why cooperation works: When User 1 is in a deep fade (blind spot), User 2 often has a clear line of sight.

* **Blue Line:** User 1 Energy (Weak).  
* **Red Line:** User 2 Energy (Strong).

*(Run phase2\_multiuser\_data.py to generate)*

### **3\. Local HMM Intelligence**

Instead of a hard threshold, each user calculates a **Log-Likelihood Ratio (LLR)**.

* **Positive LLR:** High confidence PU is ON.  
* **Negative LLR:** High confidence PU is OFF.  
* **Zero:** Uncertainty (where single nodes fail).

*(Run phase3\_hmm\_training.py to generate)*

### **4\. Performance Comparison (The Proof)**

We swept SNR from \-20 dB to \+5 dB. The **Cooperative Approach (Blue)** consistently outperforms the **Single User (Red)**, especially in the critical range of **\-10 dB to 0 dB**.

*(Run phase5\_final\_evaluation.py to generate)*

### **5\. Real-Time Dashboard**

A live simulation showing the fusion center correcting individual user errors in real-time.

*(Run phase6\_realtime\_demo.py to launch)*

## **🛠️ Technical Architecture**

1. **Signal Generation:** BPSK modulated random binary sequence.  
2. **Channel Model:** $y(t) \= h(t) \\cdot x(t) \+ n(t)$ where $h(t)$ is Rayleigh distributed.  
3. **Local Sensing:**  
   * **Feature Extraction:** Time-averaged Energy Detection.  
   * **Algorithm:** Gaussian HMM (2 states: ON/OFF).  
   * **Output:** Soft Decision (Log-Likelihood Ratio).  
4. **Data Fusion:**  
   * **Algorithm:** Bayesian Sum Rule.  
   * $\\Lambda\_{global} \= \\sum\_{i=1}^{N} \\Lambda\_{local}^{(i)}$  
   * **Decision:** Compare $\\Lambda\_{global}$ to threshold $\\gamma$.

## **💻 Installation & Usage**

### **Prerequisites**

pip install numpy matplotlib scipy hmmlearn sklearn

### **Running the Project**

The project is modular. You can run specific phases to verify different layers of the stack.

1. **Verify Physics Engine:**  
   python phase1\_environment.py

2. **Train HMMs & Check Logic:**  
   python phase3\_hmm\_training.py

3. **Generate Performance Report (ROC/SNR Curves):**  
   python phase5\_final\_evaluation.py

4. **Launch Real-Time Demo:**  
   python phase6\_realtime\_demo.py

## **👨‍💻 Tech Stack**

* **Language:** Python 3.9  
* **Math/Signal Processing:** NumPy, SciPy  
* **Machine Learning:** hmmlearn (Hidden Markov Models)  
* **Visualization:** Matplotlib (Static & Animation)

## **📜 Conclusion**

This project demonstrates that while physical signal degradation (fading) is inevitable, **algorithmic cooperation** can recover lost reliability. By moving from hard-decision energy detection to soft-decision Bayesian Fusion, we bridge the gap required for reliable Cognitive Radio networks.