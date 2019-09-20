# Installation and Running
Here is how to run the SmartPong program on your local machine!

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
