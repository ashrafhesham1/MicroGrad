import math
from graphviz import Digraph
from datetime import datetime

class Value:
    '''
    This class stores a single scalar value and its gradient and can perform back propagation 
    to construct a compuation graph up to the current instance and compute gradients with
    respect to the it. 
    
    the class implements a group of differeniable operations that include the basic mathematical
    operations along with more complex functions that are common in machine learning.
    - operations: addition / subtraction /  multiplication / division / power(int/float only)
    - functions: exp / tanh / sigmoid / relu
    '''
    def __init__(self, data, label='', _prev=(), _op=''):
        '''
        params:
            - data(int/float): the scalar value to be stored in the object
            - label(str): object label or title
            - _prev(tuple): -for internal use- holds the parents of the current object in the computation graph 
            - _op(str): -for internal use- holds the type of the operation resulted in this node
        '''
        self.data = data
        self._prev = set(_prev)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda:None
    
    def __repr__(self):
        return f'Value(data= {self.data} ,grad= {self.grad}, label= {self.label})'

    def _wrap(self, val):
        '''wrap a primitive in a Value object'''
        return Value(val) if not isinstance(val, Value) else val
    
    def __add__(self, other):
        other = self._wrap(other)
        out = Value(self.data + other.data, _prev=(self, other), _op='+')
        
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = self._wrap(other)
        out = Value(self.data * other.data, _prev=(self, other), _op='*')
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        
        return out

    def __rmul__(self, other):
        return self * other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return (-self) + other
    
    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float are supported"
        out = Value(self.data**other, _prev=(self,), _op=f'**{other}')
        
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * out.grad
        out._backward = _backward
        
        return out

    def __truediv__(self, other):
        return self * (other**-1)
    
    def __rtruediv__(self, other):
        return other * (self**-1)

    def exp(self):
        out = Value(math.exp(self.data), _prev=(self,), _op='exp')
        
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        
        return out
        
    def tanh(self):
        x = self.data
        t = (math.exp(2*x) - 1)/(math.exp(2*x) 
                                 + 1)
        out = Value(t, _prev=(self,), _op='tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out
    
    def sigmoid(self):
        x = self.data
        t = 1 / (1 + math.exp(-x))
        out = Value(t, _prev=(self,), _op='sigmoid')
        
        def _backward():
            self.grad += (out.data * (1 - out.data)) * out.grad
        out._backward = _backward
        
        return out
    
    def relu(self):
        t = max(self.data, 0)
        out = Value(t, _prev=(self,), _op='relu')
        
        def _backward():
            self.grad += 1 if out.data >=0 else 0
        out._backward = _backward
        
        return out
    
    def backward(self):
        '''perform back propagation'''
        graph, visited = [], set()
        def build_graph(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_graph(child)
                graph.append(v)
        build_graph(self)
        
        self.grad = 1.0
        for node in reversed(graph):
            node._backward()


    def _trace(self):
        ''' a helper function that builds a set of all nodes and edges in a graph for visualization '''
        nodes, edges = set(), set()

        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._prev:
                    edges.add((child, v))
                    build(child)
        build(self)

        return nodes, edges

    def plot_graph(self, save=False, title='', format='png'):
        '''plot the computation graph up to the current object'''
        dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
        nodes, edges = self._trace()
        for n in nodes:
            uid = str(id(n))
            # for any value in the graph, create a rectangular ('record') node for it
            dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
            if n._op:
              # if this value is a result of some operation, create an op node for it
              dot.node(name = uid + n._op, label = n._op)
              # and connect this node to it
              dot.edge(uid + n._op, uid)

        for n1, n2 in edges:
        # connect n1 to the op node of n2
            dot.edge(str(id(n1)), str(id(n2)) + n2._op)
        
        if save:
            title = str(datetime.timestamp(datetime.now())) if title == '' else title
            dot.render(title, format=format)
            
        return dot