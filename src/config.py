# src/config.py
class Config:
    # Meta-learning parameters
    ML_EPOCHS = 100
    ML_BATCH_SIZE = 512
    ML_LEARNING_RATE = 0.001
    
    # MTD parameters
    IP_RANGES = [(1,254), (0,254), (0,254), (0,254)]
    PORT_RANGE = (1024, 65535)
    PATH_UPDATE_INTERVAL = 5  # seconds
    
    # RL parameters
    RL_TIMESTEPS = 100000
    RL_LEARNING_RATE = 0.0003
    RL_GAMMA = 0.99
    
    # Dataset paths
    DATASETS = [
        'data/dataset_info/CIC-IDS2017',
        'data/dataset_info/CIC-DDoS2018',
        'data/dataset_info/CIC-DDoS2019',
        'data/dataset_info/CIRA-CIC-DoBrw-2020'
    ]
