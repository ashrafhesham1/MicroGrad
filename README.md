# Micrograd

A simple neural network library that operates on scalar values and consists of"
#### Engine:
This is responsible for implemnting **Back propagation** via a data structure called **Value** that implements differentiable operations and **reverse mode auto-diff** over a dynamically build **DAG**.

#### Neural Network:
this is a pytorch-like neural network API that is implemented on top of the auto-diff engine.

## Demo:
[Demo.ipynb](Demo.ipynb)

## Example Usage:
#### Auto differentiation

    w1, w2, b = Value(-3.0, label='w1'), Value(1.0, label='w2'), Value(6.8813735870195432, label='b')
    x1, x2 = Value(2.0, label='x1'), Value(0.0, label='x2')
    x1w1 = x1*w1; x1w1.label=('x1*w1')
    x2w2 = x2*w2; x2w2.label=('x2*w2')
    xw =  x1w1 + x2w2; xw.label = 'x1w1 + x2w2'
    n = xw + b; n.label = 'n'
    o = n.tanh(); o.label = 'o'
    o.backward()
    o.plot_graph()
![Auto differentiation results graph](images/autodiff.png)

#### Neural Network

    input_dim = len(xs[0])
    model = nn(input_dim, 
              layers = [
                  Dense(input_dim, 3, 'tanh', label='l1'),
                  Dense(3, 4, 'tanh',label='l2'),
                  Dense(4, 1, 'linear',label='l3')
              ], label='nn')
    loss = model.train(xs, ys, epochs=100)

![Training progress results](images/training_results.png)

    loss.plot_graph()
![Neural Network graph](images/neural_network.png)
    
**Disclaimer**: this project is greatly inspierd by the amazing tutorial by **Andrej Karpathy** with some additions and tweaks. You can find the tutorial here [micrograd](https://www.youtube.com/@AndrejKarpathy)