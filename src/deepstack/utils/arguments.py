"""
Arguments/config for DeepStack Poker neural net training (Python)
"""
class Arguments:
    def __init__(self):
        self.net = {'hidden_sizes': [128, 128]}
        self.gpu = False
        self.data_path = 'data/deepstacked_training'
        self.model_path = 'models/pretrained'
        self.epoch_count = 10
        self.learning_rate = 0.001
        self.train_batch_size = 32
