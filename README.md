# Smart-Pong
Train an AI to play Pong using reinforcement learning - Tutorial and Documentation Series
## Co-Authors
Gautam Banuru, Pranav Banuru, Ashwin Char, David Llanio, Hirad Peyvandi, Ishika Majumder

## Introduction
SmartPong is a program created by Andrej Karpathy. This program implements artificial intelligence to make it play itself and win. SmartPong outputs lines of code to see the efficiency of the AI as the number of episodes, or games, increase. 
Through this project, we learn the foundations of Artificial Intelligence by analyzing this operated program. In this project, we analyzed the Atari game called Pong, and through the reverse engineering technique, we learned how we can train an AI model to master the game pong through the deep neural network using reinforcement learning algorithm.
Figuring out how to run the program and what software and libraries were required to overcome the errors was one of the early challenges we faced. This program and many other AI programs are written in python programming language due to its simplicity. Therefore, the first application we installed was the latest version of python which is python 3.7. When you open the program, you see all these errors at the very beginning of the code where it’s asking you to install the numpy, gym, and gym(Atari) library. 

After we installed these libraries separately, in order to bring them all into one place and make use of them in our program, we installed Anaconda software which simplifies package management and enables the user to create their own environment and use a single environment to run the program. By this time, we must have had everything ready to run the program, but when we tried to do so, we were still getting an error from the gym library. After some research, we realized that the Gym library requires visual C++ Build tool, so we also had to install the visual studio for C++ to get that gym library running. Now we had all the necessary software, libraries, and packages installed and all we had to do to run the program and get some actual results was to make a few changes to the code because the code was written in Python version 2.7 and we were using python version 3.7.

## Abstract
This Technical Document serves as a guide for individuals to learn more about Artificial Intelligence, specifically Karpathy’s SmartPong. It includes a high-level description of SmartPong for advanced students and professors. We also show how to install and run Karpathy’s Smartpong code.

The version of the code being used in the analysis is the modified version. The code is broken down into mini sections and thoroughly analyzed with key concepts that are commonly used AI. We placed detailed descriptions of these key concepts prior to the analysis as a reference for readers. Finally, the conclusion proves an efficient AI is achievable and an overview of this project and our work.

## Installation and Running
Video Guide: https://www.youtube.com/playlist?list=PLwvt34ierZxIesNGya03gq5Xh48IQfj0m
### Windows 
1.	Install Python 3.7 from https://www.python.org/, make sure to add it to PATH.
2.	Install Anaconda from https://www.anaconda.com/, register Anaconda as the system Python 3.7
3.	Download the Smartpong code from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5 as a ZIP file and extract the python script.
4.	Download and install Microsoft Visual Studio from https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=Community&rel=16&rid=30015
5.	Download C++ Build tools for Visual Studio from https://visualstudio.microsoft.com/thank-you-downloading-visual-studio/?sku=BuildTools&rel=16, video recommended for this part.
6.	Download and Install Git from https://git-scm.com/download/win
7.	Look for the Anaconda prompt application in your windows search bar.
8.	Once Anaconda prompt is open type: “pip3 install gym numpy” excluding the quotation marks, press enter. This will install the gym and numpy libraries.
9.	Then type: “pip3 install git+https://github.com/Kojoley/atari-py.git” excluding the quotation marks, press enter. This will install the gym[atari] library.
10.	Find the Smartpong code in your download folder, should have the “pg.pong.py” name, right-click it and select “Edit with NotePad++”.
11.	Make the following changes:
```python
Line 3: import _pickle as pickle
Line 24: grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
Line 25: rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory
Line 43: for t in reversed(range(0, r.size)):
Line 117: for k,v in model.items():
Line 125: print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
Line 132: print ('ep %d: game finished, reward: %f' % (episode_number, reward), '' if reward == -1 else ' !!!!!!!!') 
```
12.	Save the file.
13.	Right-click the file and select “Edit with IDLE”
14.	Press run on the top bar, and then press “run module”. The program should run.
15.	Press control+c to cancel program run.

### MacOS
1.	Download and install Python 3.7 from https://www.python.org/
2.	Download and install Anaconda from https://www.anaconda.com/
3.	Download the Smartpong code from https://gist.github.com/karpathy/a4166c7fe253700972fcbc77e4ea32c5
4.	Open Anaconda and go to the environment tab, then press the create button on the bottom and create a new environment, make sure python 3.7 is selected as the package.
5.	With the new environment selected go to home and press the install button for VS code, Visual studio code.
6.	When the installation of VS code is complete, launch the application from Anaconda.
7.	Press the file tab at the top and open the Smartpong code in visual studio.
8.	Make the following changes:
Line 3: import _pickle as pickle
Line 24: grad_buffer = { k : np.zeros_like(v) for k,v in model.items() } # update buffers that add up gradients over a batch
Line 25: rmsprop_cache = { k : np.zeros_like(v) for k,v in model.items() } # rmsprop memory
Line 43: for t in reversed(range(0, r.size)):
Line 117: for k,v in model.items():
Line 125: print ('resetting env. episode reward total was %f. running mean: %f' % (reward_sum, running_reward))
Line 132: print ('ep %d: game finished, reward: %f' % (episode_number, reward), '' if reward == -1 else ' !!!!!!!!') 
9.	Under open editors right-click on the file and select “open in terminal”
10.	Type: “pip install numpy” excluding the quotation marks to install the numpy library, press enter.
11.	Type: “pip install gym” excluding the quotation marks to install the gym library, press enter.
12.	Type: “pip install gym[atari]” excluding the quotation marks to install numpy press enter.
13.	Close the terminal and save the file.
14.	Repeat step 9.
15.	Type: “python3 X.py” excluding the quotation marks where X is the name of the file, press enter. The program should run.
16.	Press control+c to cancel the program run.

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

input * weight = guess | ground_truth - guess = error | error * weight's contribution to error = adjustment

These are the three key functions of neural networks: scoring input, calculating error, and updating the model. A neural network also uses linear regression. 

y = bx + a

With many input variables, the formula looks like this:

y = b1 * x1 + b2 * x2 + b3 * x3 + a

This multiple linear regression will happen at every node of the neural network. The inputs are combined with different coefficients in different proportions.

### Policy Network
 The network that transforms input frames to output actions. The simplest method to train a policy network is with a policy gradient.

### Policy Gradient (Descent)

Gradient Descent is also known as the optimization function. Gradient is also known as "slope" and represents the relationship between the 2 variables. Because the neural network is trained to test out different weights, the relation between the error and the weights is the derivative dE/dw. This measures the slightest degree change that affects error. These are passed through an activation function, sees if there is a change in error, and checks how weights affect activation. This formula sums up the relationships.

dError/dWeight=  dError/dActivation  *dActivation/dWeight

There are numerous parameters that keep updating in order to reach peak optimization. Gradient descent is typically used when parameters cannot be calculated analytically, with algebra, but rather when an algorithm is necessary. The coefficients are exchanged in order to find the lowest and best cost. The initial values will start at 0.0 or some random small value. The derivative is calculated in order to find the rate of change of the function. Executing the derivative makes it easier to find the coefficient values by helping limit the domain. Next, α is used to define the learning rate. This specifies how much the coefficients can change on each update. If the change is too drastic, accuracy may be lost.

J(coefficient) = cost  | Δ =  d/dw cost | coefficient = coefficient - (αΔ)
	 
These steps are repeated until the cost gets as close to 0 as possible. It is recommended to plot cost vs time. By plotting the cost values per iteration, you will be able to clearly see if the gradient descent run is decreasing. If it is not decreasing, the learning rate should be reduced. 

### Rewards
The Reward Hypothesis is considered the “maximization of the expected value of the cumulative sum of a received scalar signal”. The objective is to maximize this set of rewards. In order to do this, one must look into the Markov Decision Process (MDP). 

In simple terms, the MDP states the probability of transitioning into a different state while getting some reward depending on the current state. In other words, any action in the future is only dependent on the present, not the past. Adding these rewards of different levels of significance with rewards from the future result in discounted returns. A higher discount factor leads to higher sensitivity for rewards.

### Sparse and Dense Rewards
For sparse rewards instead of receiving a reward at the end of every step/action, we receive a reward for the end of an episode. The agent learns what part of the episode action sequence led to the reward in order to imitate that sequence in the future. Sparse rewards are sample inefficient meaning the reinforcement learning program needs lots of training time before executing useful behavior. 
For a dense reward system rewards are given at the end of every step of an episode. The program is influenced by immediate rewards rather than working towards a long term goal implying a shallow lookahead. SmartPong will be using a sparse reward system.

### Discount Factor
Determines the importance of the accumulated future events in the MDP model. A higher γ leads to a higher sensitivity for rewards in future steps. To prioritize rewards in the distant future, keep gamma closer to one. A number closer to one will consider the previous results; however a number closer to zero will not learn anything. A discount factor closer to zero indicates that only rewards in the immediate future are being considered, implying a shallow lookahead. A factor equal to or greater than 1 will cause the convergence of the algorithm. However high enough discount factors (ie. γ = 0.99) result in similar intertemporal preferences with the same behavior as having a discount factor of 1. The optimal discount factor largely depends on whether a sparse reward (γ = 0.99 most optimal) or dense reward (γ = 0.8 may perform better) system is in place. If one wants to weigh earlier experience less they may try a myopic function where gamma grows linearly from 0.1 to a final gamma (ie. γ = 0.99 or 0.8) as a function of the total timesteps.

