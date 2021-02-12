from . import Simulation
import gym

gym.register(
    id="gym_quad-v0",
    entry_point = 'Quadcopter_SimCon.Simulation.gym_quad:GymQuad',
)