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

## SmartPong Roadmap

insert photo of pretty chart here

