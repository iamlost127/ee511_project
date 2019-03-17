import random

def act(delX, delY1, delY2, vel, train=False):
    print(delX, delY1, delY2, vel)
    return (random.random() < 0.1)
