# Machine Learning

Concept of training models on labelled data, testing it on unseen data, determining the error, and iterating on training to reduce error.

**Partial dervatives** are the key behind machine learning, we ask the question "*If I change x and keep all other variables constant, how does that affect my output function?*"

Typically, models contain thousands to billions of parameters which alter the models output. Taking the dervative with respect to each parameter at a time will indicate how much to tweak each individual parameter to minimize cost.

## What is Cost?

Cost is a quantity that determines how accurate a model. We aim for lower cost as this indicates a reliable and accurate model. 

We typically define a cost function, which can be defined in a variety of ways.

An example of a simple cost function: 

$$Cost = \frac{\text{Correct Predictions}}{\text{Total Predictions}}$$

## Networks and Nodes

Networks are represented by a group of nodes, often split in layers (Layer 1, Layer 2, etc..).

Nodes act similar to functions, they take in a input and produce an output.

Each node in a layer connects to every single node in the next layer using connections which have weight values. 
<div align="center">
  <img src="https://github.com/user-attachments/assets/0344486b-7686-4951-813a-a77e001ada62" height="300"/>
</div>

## Weights and Layers 

To begin, weights are usually assigned with a randomized value and get adjusted through the iterating process.

The first layer is called the Input Layer which takes in the inputs to the model. There can be an arbritrary number of hidden layers which help increase accuracy. The output layer will be the output of the model.

## Example

Let's suppose we want to determine if an individual has risk of cardiovascular disease based on two inputs, height and weight.

Our input layer will have 2 nodes which represent the height and weight respectively. 

The hidden layer will have as many nodes and layers as you wish.

The output layer will have a single node. The value can be 1 (yes they have risk) or 0 (no they don't have risk)

## Building the Network

Notice that only the input layer nodes have values. How do we calculate the other nodes in our hidden layer and eventually our result?

Each subsequent layer gets calculated by the previous layer's node values dot producted by the weights connecting them. 

<div align="center">
  <img src="https://github.com/user-attachments/assets/8a43da8b-39f0-413e-90e1-b573f31e5494" height="300"/>
</div>

We use matricies since it heavily simplifies the format rather than displaying a bunch of sums of products.

## Biases

Each node in the network, excluding nodes in the input layer, will have a bias value. 

We use this value as an addition on top of our dot product, when calculating the value for a node. 

Think of biases as allowing our model to have a y-intercept allowing for more flexibility of node values. It allows you to vertically shift your data rather than solely scaling it through weights.

## Activation Functions

Activation functions are used to squish a node's value down to something a network can deal with. 

For instance, if we wanted all the node values to be between 0 and 1, we can use a sigmoid ($\frac{1}{1+e^{-x}}$) as our activation function.

This means each node gets put through this function before the value is assigned.

## Code

Python is a great programming language for machine learning as it has numerous supported libraries.

Let's begin by initializing an array which indicates how many nodes are present in each layer.

`n = [2, 3, 3, 1]`

Now we can use the numpy library in Python to randomly generate weights and biases for our neural network.
We simply pass in the dimensions we expect our ith layer of weights to follow, which should be a *n[i] x n[i-1]* matrix

```
W1 = np.random.randn(n[1], n[0]) # Define dimensions of weights, access via the .shape property
W2 = np.random.randn(n[2], n[1])
W3 = np.random.randn(n[3], n[2])
b1 = np.random.randn(n[1], 1)
b2 = np.random.randn(n[2], 1)
b3 = np.random.randn(n[3], 1)
```

Now let's provide the input data. Using our previous example, we need to pass in two nodes (height, weight). 

Suppose X represents our input data and Y represents our output data.

```
# Passing in 10 input samples and corresponding labelled results. 
X = np.array([
    [150, 70], 
    [254, 73],
    [312, 68],
    [120, 60],
    [154, 61],
    [212, 65],
    [216, 67],
    [145, 67],
    [184, 64],
    [130, 69]
])
A0 = X.T # Transpose to allow matrix multiplcation to work (2, 10)
y = np.array([
    0, 
    1,  
    1, 
    0,
    0,
    1,
    1,
    0,
    1,
    0
])
m = 10
Y = y.reshape(n[3], m) # Reshape to reshape our data to support output layer dimensions 
```

Define our activation function
```
def sigmoid(arr):
  return 1 / (1 + np.exp(-1 * arr))
```
 
Perform the feed forward process to build our network
```
m = 10

# layer 1 calculations

Z1 = W1 @ A0 + b1  # the @ means matrix multiplication

assert Z1.shape == (n[1], m) # just checking if shapes are good
A1 = sigmoid(Z1)

# layer 2 calculations
Z2 = W2 @ A1 + b2
assert Z2.shape == (n[2], m)
A2 = sigmoid(Z2)

# layer 3 calculations
Z3 = W3 @ A2 + b3
assert Z3.shape == (n[3], m)
A3 = sigmoid(Z3)
```


