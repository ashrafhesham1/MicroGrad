import numpy as np
from autodiff.engine import Value

class Neuron:
    def __init__(self, nin, nonlin, label=''):
        self.w = [Value(np.random.uniform(-1,1), label=f'{label}_w_{i}') for i in range(nin)]
        self.b = Value(np.random.uniform(-1,1), label=f'{label}_b')
        self.nonlin = nonlin
        
    def apply_nonlin(self, val):
        non_lin_map = {
            'linear': val,
            'tanh': val.tanh(),
            'sigmoid': val.sigmoid(),
            'relu': val.relu()
        }
        
        return non_lin_map[self.nonlin]
    
    def __call__(self, x):
        output = sum([wi*xi for wi,xi in zip(self.w, x)], self.b)
        activation = self.apply_nonlin(output)
        return activation
    
    def params(self):
        return self.w + [self.b]
    
    def zero_grad(self):
        for param in self.params():
            param.zero_grad()

class Layer:
    def __init__(self) -> None:
        pass

class Dense(Layer):
    def __init__(self, nin, nunits, nonlin, label=''):
        self.nin = nin
        self.nunits = nunits
        self.units = [Neuron(nin, nonlin, label=f'{label}_n_{i}') for i in range(nunits)]
    
    def __call__(self, x):
        outs = [unit(x) for unit in self.units]
        return outs
    
    def params(self):
        return [p for unit in self.units for p in unit.params()]
    
    def zero_grad(self):
        for p in self.params():
            p.zero_grad()


class nn:
    '''Neural Network'''
    def __init__(self, nin, layers=None, layers_dim = [],  nonlin='tanh', label=''):
        '''
        params:
            - nin(int): number of inputs (feaures)
            - layers(list[Layer]): list of the layers of the network
              -- (only one of layers & layers_dim must be specified to initialize the newtwork)
            - layers_dim(list[int]): number of units of each layer in the network
              -- (only one of layers & layers_dim must be specified to initialize the newtwork)
            - nonlin(str): type of nonlinearity to be applied to the network neuron - 
              -- options: 'linear' - 'tanh' - 'sigmoid' - 'relu'  - defalut: 'tanh'
              -- require only when the network is initialized from 'layers_dim'
            - label(str): prefix to the network neurons labels
        '''

        # no initialization
        if layers is None and layers_dim == []:
            raise Exception('The network must have layers or layers_dim to initialize')
        
        # initialization from layers
        if layers is not None:
            layers_in_dim = layers[0].nunits
            layers = [Dense(nin, layers_in_dim, nonlin)] + layers if layers_in_dim != nin else layers
            dim = [layers_in_dim] + [layer.nin for layer in layers]
            
        # initialization from layer_dims
        if layers is None:
            dim = [nin] + layers_dim
            layers = [Dense(dim[i], dim[i+1], nonlin, label=f'{label}_l{i}') for i in range(len(dim)-1)]
            
        self.layers = layers
        self.dim = dim
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def params(self):
        return [p for layer in self.layers for p in layer.params()]
    
    def zero_grad(self):
        for p in self.params():
            p.grad = 0.0
    
    def _get_loss_function(self, name):
        def mse(gt, pred):
            return sum((ypred - ygt)**2 for ygt, ypred in zip(gt, pred))/len(gt)
        
        loss_functions = {
            'mse':mse
        }
        
        return loss_functions[name]
    
    def train(self, X, y, loss='mse', lr=0.01, epochs = 50, verbose=True):
        '''
        train the network on the given data

        params:
         - X(list[list[float]]): training data without the labels
         - Y(list[float]): training data labels
         - loss(str): the loss function to be used - options: 'mse' - default: 'mse'
         - lr(float): lerning rate - default: 0.01
         - epochs(int): number of epochs - default: 50
         - verbose(bool): whether or not to log the training progress - default: True
        '''
        loss_func = self._get_loss_function(loss)
        final_loss = None
        
        for epoch in range(epochs):
            #forword pass
            pred = [self(xi)[0] for xi in X]
            loss = loss_func(y, pred)
            
            #backward pass
            self.zero_grad()
            loss.backward()
            
            #update
            for p in self.params():
                p.data += p.grad * -lr
            
            # log updates
            if verbose:
                if epoch % 5 == 0:
                    print(f'finished epoch {epoch} - loss: {loss}, learning_rate: {lr}')
            
            final_loss = loss
            
        return final_loss