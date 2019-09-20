## Terms and Concepts
### Machine Learning
Machine learning is divided into three categories. Supervised learning, Unsupervised Learning, and Reinforcement Learning. In the supervised learning algorithm, we feed the data as the input to our model as well as the expected output, then we let our model come up with the fastest and the most efficient algorithm to get the desired output. In the unsupervised learning algorithm, we feed the data to our model as the input and let the algorithm to find a relationship between data and categorizes them. 

In reinforcement learning, we feed the data to our model as the input and compare the accuracy through objective function then we change the model’s parameters and repeat the operation. In other words, in reinforcement learning, we have an agent which learns how to behave in an environment by performing actions from the results. For instance, in the smartPong program, we get a snapshot of the game pong and give this data as the input to our model, our model decides whether to go up or down and finally we check whether our model has won the game to get a reward of +1 or lost the game to give a -1 reward through objective  function and based on those rewards, our model modifies its behavior through optimization algorithm. 

### Neural Networks
Neural networks contain algorithms that are modeled after the human brain in order to recognize patterns. They classify ungrouped data by finding similarities to the sample set of data. Humans group examples of data in order to be used as the “foundation” for the neural network to find patterns. Any labels that humans can identify can be used to train neural networks. Although classification requires labels, deep learning does not. 
Deep learning can also be considered as unsupervised learning. Unsupervised learning models have more chances of being accurately trained because these often train on large sets of data which help increase precision.
The networks are composed of nodes that are in each layer. Nodes assign numerical significance to input data. They also combine the input with coefficients, also known as weights. These products are then summed and passed through an activation function that determines whether or not the product will progress further. 
 
In deep neural networks, each layer of nodes trains a certain set of features based on the previous layer. This results in precision. The deeper into the neural net that you go, the more advanced elements you will see because each layer is building off the previous. When training, nodes try to reconstruct the input from samples over repetitions in order to minimize error. Neural networks find similarities and what the represent, whether it is based on labeled data or is a complete reconstruction. 
When starting to train, neural networks will create models based on a collection of weights (coefficients). This tries to model data's relationship to labels in order to grasp the data's structure. When learning, the Input is out into the network and the weights connect that input with a set of guesses. The network then takes the guess and compares it to its true classification to assume error. The weights are then adjusted to decrease error.
```python
input * weight = guess | ground_truth - guess = error | error * weight's contribution to error = adjustment
```
These are the three key functions of neural networks: scoring input, calculating error, and updating the model. A neural network also uses linear regression. 
```python
y = bx + a
```
With many input variables, the formula looks like this:
```python
y = b1 * x1 + b2 * x2 + b3 * x3 + a
```
This multiple linear regression will happen at every node of the neural network. The inputs are combined with different coefficients in different proportions.

### Policy Network
 The network that transforms input frames to output actions. The simplest method to train a policy network is with a policy gradient.

### Policy Gradient (Descent)

Gradient Descent is also known as the optimization function. Gradient is also known as "slope" and represents the relationship between the 2 variables. Because the neural network is trained to test out different weights, the relation between the error and the weights is the derivative dE/dw. This measures the slightest degree change that affects error. These are passed through an activation function, sees if there is a change in error, and checks how weights affect activation. This formula sums up the relationships.
```python
dError/dWeight=  dError/dActivation  *dActivation/dWeight
```
There are numerous parameters that keep updating in order to reach peak optimization. Gradient descent is typically used when parameters cannot be calculated analytically, with algebra, but rather when an algorithm is necessary. The coefficients are exchanged in order to find the lowest and best cost. The initial values will start at 0.0 or some random small value. The derivative is calculated in order to find the rate of change of the function. Executing the derivative makes it easier to find the coefficient values by helping limit the domain. Next, α is used to define the learning rate. This specifies how much the coefficients can change on each update. If the change is too drastic, accuracy may be lost.
```python
J(coefficient) = cost  | Δ =  d/dw cost | coefficient = coefficient - (αΔ)
```	 
These steps are repeated until the cost gets as close to 0 as possible. It is recommended to plot cost vs time. By plotting the cost values per iteration, you will be able to clearly see if the gradient descent run is decreasing. If it is not decreasing, the learning rate should be reduced. 

### Rewards
The Reward Hypothesis is considered the “maximization of the expected value of the cumulative sum of a received scalar signal”. The objective is to maximize this set of rewards. In order to do this, one must look into the Markov Decision Process (MDP). 

In simple terms, the MDP states the probability of transitioning into a different state while getting some reward depending on the current state. In other words, any action in the future is only dependent on the present, not the past. Adding these rewards of different levels of significance with rewards from the future result in discounted returns. A higher discount factor leads to higher sensitivity for rewards.

### Sparse and Dense Rewards
For sparse rewards instead of receiving a reward at the end of every step/action, we receive a reward for the end of an episode. The agent learns what part of the episode action sequence led to the reward in order to imitate that sequence in the future. Sparse rewards are sample inefficient meaning the reinforcement learning program needs lots of training time before executing useful behavior. 
For a dense reward system rewards are given at the end of every step of an episode. The program is influenced by immediate rewards rather than working towards a long term goal implying a shallow lookahead. SmartPong will be using a sparse reward system.

### Discount Factor
Determines the importance of the accumulated future events in the MDP model. A higher γ leads to a higher sensitivity for rewards in future steps. To prioritize rewards in the distant future, keep gamma closer to one. A number closer to one will consider the previous results; however a number closer to zero will not learn anything. A discount factor closer to zero indicates that only rewards in the immediate future are being considered, implying a shallow lookahead. A factor equal to or greater than 1 will cause the convergence of the algorithm. However high enough discount factors (ie. γ = 0.99) result in similar intertemporal preferences with the same behavior as having a discount factor of 1. The optimal discount factor largely depends on whether a sparse reward (γ = 0.99 most optimal) or dense reward (γ = 0.8 may perform better) system is in place. If one wants to weigh earlier experience less they may try a myopic function where gamma grows linearly from 0.1 to a final gamma (ie. γ = 0.99 or 0.8) as a function of the total timesteps.
