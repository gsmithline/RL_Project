# Final Project

## Description
This codebase is for my final project in CS 138. The project explores the Policy Space Response Oracle Framework in Normative Markov Games.

## Versions
- **Python version:** 3.10.9
- **Numpy version:** 1.23.5
- **Matplotlib version:** 3.7.0
- **Scipy version:** 3.7.0
- **Nashpy version:** 0.0.41
- **Tensorflow:** 2.15.0
- **Scikit-Learn:** 1.3.2
- **Pytorch:** 2.3.0

## Instruction for Setting up MA-Gym
Follow the instructions [here](https://github.com/koulanurag/ma-gym) to set up the necessary packages for MA-Gym:
- Focus only on the "Setup (important)" section.
- After setup, navigate to the ma-gym directory and execute `pip install -e .`. This command installs all dependencies in the environment. Note: I made significant modifications to the MA-Gym codebase, hence its included in my submission.

## Instructions to Use
Run the experiments notebook to execute the experiments.

### Notes on Running
Key hyperparameters are defined at the top of `main.py`. Please adjust these as necessary. They are currently set based on my experiment configurations, which took approximately 4 hours in total to run.

#### Hyperparameters
- **episodes:** Controls the number of episodes in the training process.
- **generations:** Specifies the number of generations in the PSRO simulation.
- **max_steps:** Determines the maximum number of steps allowed after training the agent in the environment.
- **runs:** Controls the number of simulation runs completed post-training.
