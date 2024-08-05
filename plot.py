import matplotlib.pyplot as plt
import random, time


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim((-1000, 1000))
ax.set_ylim((-1000, 1000))
ax.set_zlim((-1000, 1000))
xs, ys, zs = [], [], []
scatter = ax.scatter(xs, ys, zs)

def update():
    x = [random.randint(-1000, 1000) for x in range(10)]
    y = [random.randint(-1000, 1000) for x in range(10)]
    z = [random.randint(-1000, 1000) for x in range(10)]
    c = [random.choice(["r", "g", "b"]) for x in range(10)]
    scatter._offsets3d = (x, y, z)
    scatter.set_color(c)
    plt.pause(0.1)

while True:
    update()
    time.sleep(1)
