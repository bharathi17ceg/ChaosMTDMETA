# src/train.py
from models.meta_learning_model import MetaLearningIDS
from models.reinforcement_learning import RLAgent, MTDEnv
from src.config import Config

def train_meta_learning():
    ml_ids = MetaLearningIDS(Config.DATASETS)
    ml_ids.train(epochs=Config.ML_EPOCHS)
    ml_ids.save_model('models/meta_learning.h5')

def train_rl_agent():
    env = make_vec_env(MTDEnv, n_envs=4)
    agent = RLAgent(env)
    agent.train(timesteps=Config.RL_TIMESTEPS)
    agent.save('models/rl_agent')

if __name__ == "__main__":
    train_meta_learning()
    train_rl_agent()