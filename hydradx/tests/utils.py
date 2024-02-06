import random

def randomize_object(obj):
    if type(obj) == dict:
        return {key: randomize_object(obj[key]) for key in obj}
    elif type(obj) == list:
        return [randomize_object(item) for item in obj]
    elif type(obj) in [int, float]:
        return random.random()
    elif type(obj) == str:
        return obj + str(random.random())
    elif type(obj) == bool:
        return not obj
    elif obj is None:
        return random.random()
    else:
        for prop in obj.__dict__:
            setattr(obj, prop, randomize_object(getattr(obj, prop)))