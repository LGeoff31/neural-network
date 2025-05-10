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



 


