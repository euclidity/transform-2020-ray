import ray
ray.init(address="auto")

@ray.remote
def f(x):
    x = 0
    for burn_cycles in range(2000000): # burn more cysles to see the cluster blip
        x += burn_cycles
    return x**0.5

futures = [f.remote(i) for i in range(10000)]
print(ray.get(futures))

ray.shutdown()