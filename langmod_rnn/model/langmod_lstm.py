"""
langmod_lstm.py

Core class for defining the multi-layer LSTM-based Language Model. Consists of an embedding layer
mapping hot-encodings to the embedding space, then num_layers stacked LSTM layers, then a final
softmax layer, for predictions.
"""


class LangmodLSTM:
    def __init__(self):
        """
        Initialize a multi-layer LSTM-based language model.
        :return:
        """