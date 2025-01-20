import random

def randomize_object(obj):
    if type(obj) == dict:
        return {key: randomize_object(obj[key]) for key in obj}
    elif type(obj) == list:
        return [randomize_object(item) for item in obj]
    elif type(obj) == tuple:
        return tuple([randomize_object(item) for item in obj])
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
        return obj

def find_test_directory():
    import os

    test_dir = os.path.dirname(os.path.abspath(__file__))
    while not os.path.exists(os.path.join(test_dir, 'tests')):
        test_dir = os.path.dirname(test_dir)
        if test_dir == '/':
            raise Exception('Could not find test directory')
    return os.path.join(test_dir, 'tests')