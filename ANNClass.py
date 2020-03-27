import torch
import torch.nn.functional as F

class Net(torch.nn.Module):
    def __init__(self,**kwargs):
        super(Net,self).__init__()

        # Net accepts dictionary of inputs
        self.args = kwargs                              

        # defining RNN topology
        self.input_size  = self.args['input_size']                   # defining feature size
        self.hs1         = self.args['hs_1']                         # hidden size is equivalent to the number of nodes in hidden layers
        self.hs2         = self.args['hs_2']                         # number of layers
        self.output_size = self.args['output_size']                  # defining output size 

        # defining neural architecture
        self.hidden1    = torch.nn.Linear(self.input_size, self.hs1)      # neurons in hidden layer1
        self.hidden2    = torch.nn.Linear(self.hs1, self.hs2)             # neurons in hidden layer2

        # defining output layer
        self.p_mu       = torch.nn.Linear(self.hs2, self.output_size)     # mean outputs
        self.p_std      = torch.nn.Linear(self.hs2, self.output_size)     # std outputs
        
    def forward(self, inputs):

        x = inputs 
        z = torch.tanh(self.hidden1(x))             # activation function of hidden layer 1
        y = torch.tanh(self.hidden2(z))             # activation function of hidden layer 2
        
        mu  = self.p_mu(y)                          # linear output layer for mean prediction
        std = self.p_std(y)                         # linear output layer for std prediction
        
        mu      = torch.sigmoid(mu)                 # activation layer
        std     = torch.sigmoid(std)                # activation layer

        return (mu, std)

    def weights_init(self,m):
        # function to reset weights after each fold
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)



