from numpy.random import default_rng

rng = default_rng()
numbers = rng.choice(20, size=10, replace=False)
print(numbers)
