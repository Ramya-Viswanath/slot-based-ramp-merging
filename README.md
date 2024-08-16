# slot-based-ramp-merging
The project trains RL agent using PPO to merge a single vehicle at the given time slot and desired speed.

To install required dependencies
python install -r requirements.txt

Scenario 1: Discrete action space
cd Discrete_action_space folder , python main.py to train the single agent with discrete action space.

Scenario 2.1 : Learning rate experiment
cd Learning Rate Variation
python main.py -> Trains the agent for various learning rates and saves the model, tests the model for 5000 episodes for different learning rates.
python test.py -> Tests the model fopr 5 runs for each learning rate.

Scenario 2.2 : Discount Rate experiment
cd Discount Factor Variation
python main.py -> Trains the agent for various discount rates and saves the model, tests the model for 5000 episodes for different discount rates.
python test.py -> Tests the model fopr 5 runs for each discount factor.

Scenario 2.3 : Continuous action space
The obtained optimal values of hyper-parameters is used for the final training and testing.
cd Continuous_action_space
python main.py -> training
python test.py -> testing

Scenario 3 : Varying Initial speed
cd varying Initial speed
python main.py -> training
python test.py -> testing

Scenario 4 : Varying merge speed
cd Varying merge speed
python main.py -> training
