#Kyle Norland
#Date: 1/30/23
#Purpose: Plot the evolution of the Q-table over time

import matplotlib.pyplot as plt
from matplotlib import interactive
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation
import time

def update_plot(input_array):
    global surf
    
    #Generate data
    rows = len(input_array)
    columns = len(input_array[0])
    X = np.arange(0, columns, 1)
    Y = np.arange(0, rows, 1)
    X, Y = np.meshgrid(X, Y)
    
    #Remove existing surface
    surf.remove()
    
    #Plot the surface.
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
                           
    print("Showing plot")
    plt.draw()
    fig.canvas.flush_events()
    time.sleep(1)
    '''
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    '''
'''
# setup the figure and axes
fig = plt.figure(figsize=(8, 3))
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')

# fake data
_x = np.arange(4)
_y = np.arange(5)
_xx, _yy = np.meshgrid(_x, _y)
x, y = _xx.ravel(), _yy.ravel()

top = x + y
bottom = np.zeros_like(top)
width = depth = 1

ax1.bar3d(x, y, bottom, width, depth, top, shade=True)
ax1.set_title('Shaded')

ax2.bar3d(x, y, bottom, width, depth, top, shade=False)
ax2.set_title('Not Shaded')

plt.show()

'''
#Put initial matplotlib setting
interactive(True)

#Generate initial plot
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(0, 6, 1)
Y = np.arange(0, 5, 1)
X, Y = np.meshgrid(X, Y)
Z = np.random.random((5,6))

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.draw()
fig.canvas.flush_events()
time.sleep(2)

#Update the plot to form the animation
for i in range(2):
    Z = np.random.random((5,6))
    update_plot(Z)

#Alt update
'''
line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
                                   interval=50, blit=False)
'''