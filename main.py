"""
IEEE Paper Reproduction: Probabilistic Forecasting of Residential Load via Transfer Learning
Entry point of the experiment.
"""

import torch
from torch.utils.data import DataLoader
import os
import sys


from src.dataset import EnedisDataProcessor, LoadDataset
from src.model import Model
from src.trainer import Trainer
from src.utils import Metrics, plot_all_results

import torch

# ============================ Hyperparameter Configuration ============================
CONFIG = {
    # ------------------- 1. Global & Environment Settings -------------------
    # Pipeline control: 
    # True  -> Execute the full Two-Stage pipeline (Source Pre-training + Target Fine-tuning).
    # False -> Load existing weights from checkpoints for direct inference/evaluation.
    "TRAIN_MODE": True,             
    
    # Hardware accelerator selection (Automatically fall back to CPU if no GPU is found)
    "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    
    # Path to the dataset (Aggregated residential load data)
    "CSV_PATH": 'data/conso-inf36-region.csv',
    
    # ------------------- 2. Sequence Formatting (Sliding Window) ------------
    # Lookback window size: 
    # 12 steps * 30-min resolution = 6 hours of historical data to capture short-term trends.
    "SEQ_LEN": 12,                  
    
    # Prediction horizon: 
    # Single-step forecasting. Predicting the load for the next 1 step (next 30 mins).
    "PRED_LEN": 1,                  
    
    # Number of samples per training batch
    "BATCH_SIZE": 256,
    
    # ------------------- 3. Transfer Learning Strategy ----------------------
    # Offline Training Stage (Source Domain):
    # Maximum epochs to train on the data-rich source domain to extract general knowledge.
    "EPOCHS_PRE": 15,               
    
    # Fine-tuning Stage (Target Domain / Deficient Data):
    # Epochs to adapt the frozen model to the limited target domain data.
    "EPOCHS_FT": 10,                
    
    # Learning rate for source domain pre-training
    "LR_PRE": 1e-3,                 
    
    # Learning rate for target domain fine-tuning. 
    # Must be strictly smaller than LR_PRE to prevent catastrophic forgetting 
    # and variance collapse in the small-sample target domain.
    "LR_FT": 1e-4,                  
    
    # Early stopping patience mechanism to prevent overfitting in both stages
    "PATIENCE": 3,
    
    # ------------------- 4. Model Architecture (Attention-LSTM) -------------
    # Dimension of the hidden state vectors in the LSTM layers
    "HIDDEN_DIM": 128,              
    
    # Number of parallel attention heads (Eq. 10) to capture diverse temporal dependencies
    "N_HEADS": 4,                   
    
    # Number of stacked LSTM layers to enhance non-linear temporal representation
    "N_LAYERS": 2,                  
    
    # ------------------- 5. Probabilistic Forecasting (MVE) -----------------
    # The critical value (Z-score) for constructing the Prediction Interval (PI).
    # 1.96 corresponds to a 95% Confidence Level under a Gaussian assumption (Eq. 14).
    "Z_SCORE": 1.96                 
}

if __name__ == '__main__':
    # ========================== Initialization Header ==========================
    print("="*85)
    print("   This repository contains a high-fidelity PyTorch implementation of the paper: ")
    print("   A Transfer Learning-based Method for Probabilistic Forecasting of Residential ")
    print("   Load with Deficient DataPublished in IEEE Transactions on Smart Grid*")
    print("-"*85)
    print("   Author:      Ziyang Wang")
    print("   Affiliation: College of Electrical Engineering, Sichuan University")
    print("   Date:        March 2026")
    print("="*85)
    print(f"\n[Environment] Running on: {CONFIG['DEVICE']}")
    print(f"[Dataset Path] {CONFIG['CSV_PATH']}\n")
    # ===========================================================================

    # ------------------- 1. Data Pipeline Initialization & Data Loading -------------------
    # Core step: Initialize the dedicated data processor and load the raw dataset

    # Initialize the Enedis electricity load data processor
    # Set sequence length and prediction length according to configuration parameters
    proc = EnedisDataProcessor(seq_len=CONFIG["SEQ_LEN"], pred_len=CONFIG["PRED_LEN"])

    # Load raw CSV dataset from the specified path in the configuration
    df = proc.load_data(CONFIG["CSV_PATH"])

    # Print a separator and title for data sample preview
    # Print the preview information to verify the correctness of data loading and format
    print("\n" + "~"*30 + " Data Sample Preview (First 5 Rows) " + "~"*30)

    # Import pandas library and set display options
    # Show all columns and expand display width for clear data viewing
    import pandas as pd; pd.set_option('display.max_columns', None); pd.set_option('display.width', 1000)

    # Print the first 5 rows of the loaded DataFrame for quick data validation
    print(df.head(5))

    # Print end separator for the data preview section
    print("~"*80)

    # ------------------- Data Sequence Construction & Domain Splitting -------------------
    
    # Step 2: Transform the processed tabular dataframe into 3D time-series sequences.
    # X: Input feature tensor [Batch, SEQ_LEN, Features]
    # y: Target label for the next time step [Batch]
    # gids: Group identifiers used to distinguish between Source and Target domains.
    X, y, gids = proc.build_sequences(df)

    # Step 3: Split the dataset into Source Domain (data-rich) and Target Domain (data-deficient).
    # Returns a tuple of 5 datasets: (src_train, src_val, tgt_train, tgt_val, tgt_test)
    # This step is critical for simulating the "Deficient Data" scenario in the target domain.
    splits = proc.split(X, y, gids)

    # ------------------- PyTorch Data Pipeline Initialization -------------------
    
    # Encapsulate raw NumPy arrays into PyTorch DataLoaders for efficient batch processing.
    # Data is automatically migrated to the designated hardware (GPU/CPU) via LoadDataset.
    dls = {
        # Source Domain DataLoaders: Utilized for the "Offline Pre-training Stage" 
        # to capture general residential electricity consumption patterns.
        'src_tr': DataLoader(
            LoadDataset(splits[0][0], splits[0][1], CONFIG["DEVICE"]), 
            CONFIG["BATCH_SIZE"], 
            shuffle=True  # Shuffle to improve the robustness of the pre-trained feature extractor
        ),
        'src_val': DataLoader(
            LoadDataset(splits[1][0], splits[1][1], CONFIG["DEVICE"]), 
            CONFIG["BATCH_SIZE"]
        ),

        # Target Domain DataLoaders: Utilized for the "Fine-tuning Stage" (tgt_tr/tgt_val)
        # and the "Final Performance Evaluation" (tgt_te).
        'tgt_tr': DataLoader(
            LoadDataset(splits[2][0], splits[2][1], CONFIG["DEVICE"]), 
            CONFIG["BATCH_SIZE"], 
            shuffle=True  # Fine-tune the Dense head on the few-shot target samples
        ),
        'tgt_val': DataLoader(
            LoadDataset(splits[3][0], splits[3][1], CONFIG["DEVICE"]), 
            CONFIG["BATCH_SIZE"]
        ),
        'tgt_te': DataLoader(
            LoadDataset(splits[4][0], splits[4][1], CONFIG["DEVICE"]), 
            CONFIG["BATCH_SIZE"] # Testing on the held-out target dataset to verify PI metrics
        )
    }

# ============================ 2.Model & Trainer Initialization ============================
    # Step 2.1: Instantiate the Hybrid Attention-LSTM Probabilistic Forecaster.
    # This model implements the architecture described in Fig. 2 of the original paper.
    # It combines feature weighting, temporal extraction (LSTM), and importance focus (Attention).
    model = Model(
        in_dim=X.shape[-1],             # Input dimension: load history + temporal + static features
        hidden_dim=CONFIG["HIDDEN_DIM"],# Capacity of the LSTM and Attention feature space
        n_heads=CONFIG["N_HEADS"],      # Parallel attention heads to capture diverse periodic patterns (Eq. 10)
        n_layers=CONFIG["N_LAYERS"]     # Stacks of LSTM to model high-order non-linear temporal dynamics
    )

    # Initialize the trainer module responsible for model training, fine-tuning, validation, and saving
    # Manages the entire optimization process: pre-training, fine-tuning, early stopping, and checkpoint storage
    trainer = Trainer(
        model,                          # The initialized prediction model
        CONFIG["DEVICE"],               # Computing device (GPU if available, otherwise CPU)
        CONFIG["LR_PRE"],               # Learning rate for source domain pre-training
        CONFIG["LR_FT"],                # Learning rate for target domain fine-tuning
        CONFIG["EPOCHS_PRE"],           # Total epochs for offline pre-training
        CONFIG["EPOCHS_FT"],            # Total epochs for few-shot fine-tuning
        CONFIG["PATIENCE"],             # Patience value for early stopping (prevent overfitting)
        "checkpoints"                   # Directory to save the best model weights during training
    )
# ------------------- 3.Core Training & Knowledge Transfer Workflow -------------------

    # Conditional control to switch between a full training pipeline or a quick evaluation.
    if CONFIG["TRAIN_MODE"]:
        
        # Phase 1: Source Domain Pre-training (The "Offline Training Stage").
        # Training the full model (Attention-LSTM + MVE Head) on the data-rich source domain.
        # The objective is to learn generalized residential load patterns and common
        # periodic consumption behaviors (Day/Week trends) from large-scale datasets.
        trainer.pretrain(dls['src_tr'], dls['src_val'])
        
        # Phase 2: Target Domain Adaptation (The "Fine-tuning Stage").
        # Addressing the "Deficient Data" problem. Here, the pre-trained feature extractor 
        # (Weighting layer, LSTM, and Attention) is frozen to preserve general knowledge.
        # Only the MVE output head is updated using the few-shot samples from the target domain 
        # to adapt the probability distribution (mu and sigma) to local user characteristics.
        trainer.finetune(dls['tgt_tr'], dls['tgt_val'])
        
    else:
        # ------------------- Inference & Deployment Mode -------------------
        # This branch is triggered when the user wants to bypass the computationally 
        # expensive pre-training phase and use existing optimized weights.
        
        print("\n[Mode: Inference/Evaluation]: Loading pre-trained state for target domain deployment.")
        
        import os
        # Verify and load the "Best Source Model" checkpoint.
        # This file contains the feature representation parameters that achieved the 
        # lowest NLL loss on the source validation set.
        if os.path.exists(trainer.best_ckpt): 
            trainer.m.load_state_dict(
                torch.load(trainer.best_ckpt, map_location=CONFIG["DEVICE"])
            )
            print(f"Successfully reloaded best source weights from: {trainer.best_ckpt}")
        else:
            print("Error: No pre-trained checkpoint found. Proceeding with initial parameters.")
            sys.exit()

    # ------------------- 4.Probabilistic Inference & Result Retrieval -------------------

    # Step 4: Perform probabilistic forecasting on the unseen target domain test set (tgt_te).
    # This phase evaluates the model's final generalization performance after the 
    # two-stage transfer learning process. 

    # The predict() method executes the forward pass and returns two result dictionaries:
    # 1. res_norm: Contains [mu, lo, hi, y] in the normalized feature space (scaled).
    #    - Essential for calculating dimensionless metrics to benchmark against 
    #      the results in Table 1 of the original paper.
    # 2. res_real: Contains the same variables mapped back to physical units (Wh).
    #    - Necessary for analyzing actual energy consumption and rendering the 
    #      visual prediction intervals in the final figures.

    res_norm, res_real = trainer.predict(
        dl=dls['tgt_te'],           # The held-out target test set (never seen during training)
        scaler=proc.scaler,         # The fitted StandardScaler used for inverse transformation
        z_score=CONFIG["Z_SCORE"]   # Statistical multiplier (1.96) applied to Eq. (14) 
                                    # to construct the 95% Prediction Interval (PI)
    )
    
    # ------------------- 5. Performance Metric Evaluation -------------------
    # Calculate and print core evaluation metrics for probabilistic forecasting
    # Two sets of metrics are computed for standardized space and real physical space respectively

    # Print header for metrics in normalized feature space
    # These metrics are used for strict comparison with the baseline methods in Paper Table 1
    print("\n===== Evaluation Metrics (Normalized Space - Strictly Aligned with Paper Table 1) =====")
    # Calculate forecasting metrics using normalized predictions and ground truth
    # Input parameters: predicted mean, prediction interval bounds, ground truth, and confidence coefficient
    m_norm = Metrics.calculate(res_norm['mu'], res_norm['lo'], res_norm['hi'], res_norm['y'], CONFIG["Z_SCORE"])
    # Print all metrics with 6 decimal places for high-precision academic comparison
    for k, v in m_norm.items(): 
        print(f"  {k:<8}: {v:.6f}")

    # Print header for metrics in real physical space (Unit: Watt-hours, Wh)
    # These metrics reflect the actual forecasting performance in real-world engineering scenarios
    print("\n===== Evaluation Metrics (Real Physical Space - Wh) =====")
    # Calculate forecasting metrics using real-scale predictions and ground truth
    m_real = Metrics.calculate(res_real['mu'], res_real['lo'], res_real['hi'], res_real['y'], CONFIG["Z_SCORE"])
    # Print all metrics with 4 decimal places for practical engineering application display
    for k, v in m_real.items(): 
        print(f"  {k:<8}: {v:.4f}")

# ------------------- 6. Result Visualization & Figure Rendering -------------------
    # Visualize all experimental results and generate high-resolution figures for paper writing
    # Persist the generated plots to the local directory for result demonstration

    # Print execution progress: rendering and saving high-definition figures
    print("\n[6/6] Rendering and persisting high-definition paper figures...")

    # Invoke the unified plotting function to generate complete experimental visualization
    # Parameters Explanation:
    # res_real: Prediction results in real physical units (Wh) for intuitive result display
    # (trainer.pretrain_train_loss, trainer.pretrain_val_loss): Loss curves of source domain pre-training stage
    # (trainer.finetune_train_loss, trainer.finetune_val_loss): Loss curves of target domain fine-tuning stage
    plot_all_results(res_real, 
                    (trainer.pretrain_train_loss, trainer.pretrain_val_loss), 
                    (trainer.finetune_train_loss, trainer.finetune_val_loss))

    # Print experiment completion prompt
    # All figures are saved in the img/ directory for paper illustration
    print("Experiment completed perfectly! Figures have been saved to the img/ directory.")
