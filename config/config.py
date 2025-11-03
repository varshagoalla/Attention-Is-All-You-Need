import torch

class Config:
    train_src = "../data/train.src"
    train_tgt = "../data/train.tgt"
    val_src = "../data/val.src"
    val_tgt = "../data/val.tgt"
    model_path = "../data/wmt_bpe.model"

    batch_size = 64
    lr = 3e-4
    num_epochs = 10
    save_dir = "./checkpoints"
    num_workers = 4

    d_model = 512
    n_heads = 8
    num_layers = 6
    dropout = 0.1
    ff_dim = 2048

    device = "cuda" if torch.cuda.is_available() else "cpu"
