from dqn.main import cart_pole_env

SEED = 0x1337_b00b

def hello() -> str:
    cart_pole_env(seed=None)
    return "Hello from dqn!"
