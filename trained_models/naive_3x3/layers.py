from tictactoe_nn.layers import ConvolutionLayer, ActivationLayer


layers_def = [
    ConvolutionLayer(1, 8, (3, 3)),
    ActivationLayer('relu', 8),
    ConvolutionLayer(8, 16, (3, 3)),
    ActivationLayer('relu', 16),
    ConvolutionLayer(16, 1, (3, 3)),
    ActivationLayer('sigmoid', 1),
]
