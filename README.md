My fork of the warmup repo

Intended for porting from gym==0.13 ---> gymnasium==0.29.1

`main.py`
```py
import gymnasium as gym
import warmup

env = gym.make("humanreacher-v0", render_mode="human")

observation, info = env.reset(seed=69)
for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(info)
    print(action.shape)
    print(observation.shape)

    if terminated or truncated:
        observation, info = env.reset()

env.close()
```

# warmup
Gym environments for musculoskeletal reaching tasks.
More detailed README will follow.

## Environments
Available environments are:
`muscle_arm-v0` `torque_arm-v0`, `humanreacher-v0`

## Example code

```
import gym
import warmup

env = gym.make("humanreacher-v0")

for ep in range(5):
     ep_steps = 0
     state = env.reset()
     while True:
         next_state, reward, done, info = env.step(env.action_space.sample())
         env.render()
         if done or (ep_steps >= env.max_episode_steps):
             break
         ep_steps += 1
```
