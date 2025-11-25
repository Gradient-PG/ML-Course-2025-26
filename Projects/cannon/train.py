# Exemplary training script
# Model should be able to hit the target reliably within first two shots
# it can modify the strength and angle of the cannon in each iteration

# Reinforcement learning is the suggested way of solving this,
# try to base the reward of the distance to the target (result[0])

from cannon import CannonEnv
import torch

cannon_env = CannonEnv()
calibration = torch.tensor([0, 0])

for i in range(1000):
    result = cannon_env.fire(calibration[0], calibration[1])

    # model.learn(result)
    # calibration = model.predict(result)

    if cannon_env.is_target_hit():
        cannon_env = CannonEnv()
