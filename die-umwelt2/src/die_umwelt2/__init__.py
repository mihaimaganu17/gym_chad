from gymnasium.envs.registration import register

register(
    # The environment ID consists of three components, two of which are optional:
    # - An optional namespace: `gymnasium_env`
    # - A mandatory name: `GridWorld`
    # - An optional, but recommended version: `v0`
    id="dieumwelt/GridWorld-v0",
    entry_point="die_umwelt2.envs:GridWorldEnv",
)

def install():
    print("yes")