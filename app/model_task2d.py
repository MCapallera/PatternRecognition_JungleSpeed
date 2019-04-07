"""
source activate deepdiva
python template/RunMe.py --dataset-folder download_datasets/MNIST/ --output-folder results_PatRec/ --model-name PR_MLP2B --ignoregit --lr 0.1 --epochs 40 --experiment-name jungeSpeed2d 2>/dev/null &
python template/RunMe.py --dataset-folder download_datasets/permutatedmnist/ --output-folder results_PatRec/ --model-name PR_MLP2B --ignoregit --lr 0.1 --epochs 40 --experiment-name jungeSpeed2d 2>/dev/null &

python template/RunMe.py --dataset-folder download_datasets/MNIST/ --output-folder results_PatRec/ --model-name PR_CNN --ignoregit --lr 0.1 --epochs 35 --experiment-name jungeSpeed2d 2>/dev/null &
python template/RunMe.py --dataset-folder download_datasets/permutatedmnist/ --output-folder results_PatRec/ --model-name PR_CNN --ignoregit --lr 0.1 --epochs 35 --experiment-name jungeSpeed2d 2>/dev/null &

"""

import torch.nn as nn


class Flatten(nn.Module):
    """
    Flatten a convolution block into a simple vector.

    Replaces the flattening line (view) often found into forward() methods of networks. This makes it
    easier to navigate the network with introspection
    """
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class PR_MLP2B(nn.Module):
    """
    Basic MLP port

    Attributes
    ----------
    expected_input_size : tuple(int,int)
        Expected input size (width, height)
    conv1 : torch.nn.Sequential
    conv2 : torch.nn.Sequential
    conv3 : torch.nn.Sequential
        Convolutional layers of the network
    fc : torch.nn.Linear
        Final classification fully connected layer

    """

    def __init__(self, **kwargs):
        """
        Creates an CNN_basic model from the scratch.

        Parameters
        ----------
        output_channels : int
            Number of neurons in the last layer
        input_channels : int
            Dimensionality of the input, typically 3 for RGB
        """
        super(PR_MLP2B, self).__init__()

        # PR_FILL_HERE: Here you have to put the expected input size in terms of width and height of your input image
        self.expected_input_size = (28, 28)

        self.mlp = nn.Sequential(
            Flatten()
            , nn.Linear(784*3, 62)
            , nn.ReLU()
            , nn.Linear(62, 10)
            , nn.Softmax()
        )

    def forward(self, x):
        """
        Computes forward pass on the network

        Parameters
        ----------
        x : Variable
            Sample to run forward pass on. (input to the model)

        Returns
        -------
        Variable
            Activations of the fully connected layer
        """
        # flatten the input
        x = self.mlp(x)
        return x
