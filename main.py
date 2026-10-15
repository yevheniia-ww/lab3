import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def holder(x, y):
    radius = np.sqrt(x*x + y*y)
    return -abs(np.sin(x)*np.cos(y)*np.exp(abs(1-radius/np.pi)))


np.random.seed(7)
generations = 80
population_size = 50
areas = [(-10, 0, -10, 0),
         (-10, 0, 0, 10),
         (0, 10, -10, 0),
         (0, 10, 0, 10)]
colors = ['red', 'dodgerblue', 'orange', 'lime']
populations = []
history = []
best_values = [[] for area in areas]

for left, right, bottom, top in areas:
    x = np.random.uniform(left, right, population_size)
    y = np.random.uniform(bottom, top, population_size)
    populations.append(np.column_stack((x, y)))

history.append([group.copy() for group in populations])

for generation in range(generations):
    for area_index in range(4):
        population = populations[area_index]
        values = holder(population[:, 0], population[:, 1])
        population = population[np.argsort(values)]
        best_values[area_index].append(np.min(values))

        elite = population[:5].copy()
        parents = population[:20]
        children = []
        left, right, bottom, top = areas[area_index]

        while len(children) < population_size - 5:
            first, second = parents[np.random.randint(20, size=2)]
            mix = np.random.rand()
            child = mix*first + (1-mix)*second

            if np.random.rand() < 0.5:
                child += np.random.normal(0, 0.9*(1-generation/generations)+0.03, 2)

            child[0] = np.clip(child[0], left, right)
            child[1] = np.clip(child[1], bottom, top)
            children.append(child)

        populations[area_index] = np.vstack((elite, children))

    history.append([group.copy() for group in populations])

print('Знайдені мінімуми:')
for area_index in range(4):
    population = populations[area_index]
    values = holder(population[:, 0], population[:, 1])
    best_index = np.argmin(values)
    x, y = population[best_index]
    print(f'Область {area_index+1}: x = {x:.5f}, '
          f'y = {y:.5f}, f = {values[best_index]:.5f}')

x = np.linspace(-10, 10, 250)
y = np.linspace(-10, 10, 250)
grid_x, grid_y = np.meshgrid(x, y)
grid_values = holder(grid_x, grid_y)

figure, axes = plt.subplots(figsize=(9, 8))
axes.contourf(grid_x, grid_y, grid_values, 40, cmap='viridis')
points = []
for color in colors:
    points.append(axes.scatter([], [], s=15, c=color))

for min_x in (-8.05502, 8.05502):
    for min_y in (-9.66459, 9.66459):
        axes.plot(min_x, min_y, 'y*', ms=16,
                  markeredgecolor='black')

axes.set(xlim=(-10, 10), ylim=(-10, 10),
         xlabel='x', ylabel='y')


def show_generation(generation):
    for area_index in range(4):
        points[area_index].set_offsets(history[generation][area_index])
    axes.set_title(f'Покоління {generation} / {generations}')
    return points


animation = FuncAnimation(figure, show_generation,
                          frames=len(history), interval=100)
animation.save('evolution.gif', writer=PillowWriter(fps=10))
show_generation(generations)
figure.savefig('result.png', dpi=160)
plt.close(figure)

plt.figure(figsize=(9, 5))
for area_index in range(4):
    plt.plot(best_values[area_index], colors[area_index],
             label=f'Область {area_index+1}')
plt.xlabel('Покоління')
plt.ylabel('Найкраще значення f(x, y)')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('progress.png', dpi=160)
plt.close()
