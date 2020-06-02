import roboschool
import gym

env = gym.make('RoboschoolAnt-v1')
env.reset()
for _ in range(100):
    print(env.step(env.action_space.sample()))
    #TODO:
    #rgb = env.render('rgb_array')
    #print(rgb)

