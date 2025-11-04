import torch

class Config:
    # Data paths
    train_src = "data/train.en"
    train_tgt = "data/train.de"
    val_src = "data/val.en"
    val_tgt = "data/val.de"
    model_path = "data/wmt_bpe.model"

    # Model architecture parameters
    d_model = 512
    n_heads = 8
    num_layers = 6
    dropout = 0.1
    ff_dim = 2048
    max_len = 5000

    # Training parameters
    batch_size = 32 # Paper uses 25000 tokens per batch which means variable batch size
    num_epochs = 20
    num_workers = 4
        
    # Optimizer parameters
    lr = 1.0 
    beta1 = 0.9
    beta2 = 0.98
    eps = 1e-9
        
    # Scheduler parameters
    warmup_steps = 4000
        
    # Loss parameters
    label_smoothing = 0.1
        
    # Gradient clipping
    clip_grad = 1.0
        
    # Checkpoint directory
    checkpoint_dir = 'checkpoints'

    # Device configuration
    device = "cuda" if torch.cuda.is_available() else "cpu"
