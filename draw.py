import argparse
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from src.IoTDevice import *
from src.Environment import Environment
from utils import read_config
from src.Map.Map import *
from src.Display import *
import matplotlib.animation as animation
from functools import partial
import ffmpeg
# Install ffmpeg, python 3.7.16 and the rest just as mentioned in the PPT

def TrapsIndex(map,agents):
    if(map == "urban50"):
        traps = [[0,47],[16,16],[15,25],[0,37],[6,34]]
    else:
        if(agents==2):
            traps = [[0,3],[16,16],[15,0],[29,25],[28,15]]
        else:
            traps = [[8,6], [0,27], [13,16], [19,25], [23,22]]
    return traps
def DeviceIndex(map,agents):
    if(map == "urban50"):
        device_p = [[16,9], [13,14], [4,24], [13,40], [18,44], [23,44], [37,40], [44,20]]
        device_color = ['purple','green', 'grey', 'red', 'blue', 'orange', 'brown', 'pink']
    else:
        if(agents==2):
            device_p = [[7,22], [10,29], [9,8], [16,6], [25,10], [29,4], [29, 8],[29,10],[28,16]]
            device_color = ['grey', 'red', 'pink','blue', 'orange', 'green','purple','brown','olive']
        else:
            device_p = [[1,2],[4,11],[11,27],[19,28],[18,5], [24,11]]
            device_color = ['purple', 'orange', 'brown', 'green', 'red', 'blue']
    return device_p,device_color
def TrapChecker(map,agents,X,Y):
    traps = TrapsIndex(map[0],agents[0])
    index = -1
    XY = []
    for i in traps:
        if(X==i):
            index = i
            XY = X
        if(Y==i):
            index = i
            XY = Y
    return index, XY
def drawTrapped(XY):
    ax = plt.gca()
    ax.add_patch(patches.Rectangle([XY[0]-0.5,XY[1]], 1, 1, fill=False, edgecolor='red', color='red', linewidth='0.8', hatch='x'))
    ax.add_patch(patches.Rectangle([XY[0],XY[1]-0.5], 1, 1, fill=False, edgecolor='red', color='red', linewidth='0.8', hatch='x'))
    ax.add_patch(patches.Rectangle([XY[0]-0.5,XY[1]-0.5], 1, 1, fill=False, edgecolor='red', color='red', linewidth='0.8', hatch='x'))
    ax.add_patch(patches.Rectangle([XY[0]+0.5,XY[1]], 1, 1, fill=False, edgecolor='red', color='red', linewidth='0.8', hatch='x'))
    ax.add_patch(patches.Rectangle([XY[0],XY[1]+0.5], 1, 1, fill=False, edgecolor='red', color='red', linewidth='0.8', hatch='x'))
    ax.add_patch(patches.Rectangle([XY[0]+0.5,XY[1]+0.5], 1, 1, fill=False, edgecolor='red', color='red', linewidth='0.8', hatch='x'))
    ax.add_patch(patches.Rectangle([XY[0]-0.5,XY[1]+0.5], 1, 1, fill=False, edgecolor='red', color='red', linewidth='0.8', hatch='x'))
    ax.add_patch(patches.Rectangle([XY[0]+0.5,XY[1]-0.5], 1, 1, fill=False, edgecolor='red', color='red', linewidth='0.8', hatch='x'))
def RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents):
    map_path = 'res/' + map + '.png'
    TITLE = model + " with " + str(agents) +" agents - " + map
    print(TITLE)
    
    env_map = load_map(map_path)
    dis = DHDisplay()
    fig_size = 5.5
    fig = plt.figure(figsize=[fig_size, fig_size])
    ax_traj = plt.axes()
    device_p, device_color = DeviceIndex(map,agents)
    for i in range(len(device_p)):
        ax_traj.add_patch(patches.Circle(np.array(device_p[i]) + np.array((0.5, 0.5)), 0.6, facecolor=device_color[i],edgecolor=None))
    traps = TrapsIndex(map,agents)
    for trap in traps:
        ax_traj.add_patch(plt.Rectangle(trap, 1, 1, fill=False, edgecolor='white', color='purple', linewidth='0.8', hatch='///////'))
    value_map = np.ones(env_map.get_size(), dtype=float)
    dis.create_grid_image(ax=ax_traj, 
                        env_map=env_map, 
                        value_map=value_map)
    I=[]
    for i in range(max(len(X1),len(X2),len(X3))):
        I.append(i)
    ani = animation.FuncAnimation(fig, animateAgent, frames=I, fargs=([map],[agents]) ,repeat=False, blit=False)
    plt.title(TITLE)
    fig.tight_layout()
    writefig = animation.PillowWriter(fps=20)
    ani.save("RealtimeDraw/" + model + map + "-Agents" + str(agents) + ".gif",writer=writefig)
def animateAgent(i,map,agents):
    dis = DHDisplay()
    try:
        dis.draw_movement(from_position=X1[i],to_position=Y1[i],color='green')
        if(i == len(X1)-1):
            index, XY = TrapChecker(map,agents,X1[::-1][0],Y1[::-1][0])
            if(index != -1):
                drawTrapped(XY)
    except:
        print("Agent1 Finished")
    try:
        dis.draw_movement(from_position=X2[i],to_position=Y2[i],color='black')
        if(i == len(X2)-1):
            index, XY = TrapChecker(map,agents,X2[::-1][0],Y2[::-1][0])
            if(index != -1):
                drawTrapped(XY)
    except:
        print("Agent2 Finished")
    try:
        dis.draw_movement(from_position=X3[i],to_position=Y3[i],color='blue')
        if(i == len(X3)-1):
            index, XY = TrapChecker(map,agents,X3[::-1][0],Y3[::-1][0])
            if(index != -1):
                drawTrapped(XY)
    except:
        print("Agent3 Finished")
#############################################
# Our Model
#############################################
####### manhattan32
map = "manhattan32"
model = "MPDA"
draw = 0
if draw==1:
    # Manhattan32
    #single-drone:
    agents = 1
    X1 = [[23, 28], [22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [10, 28], [9, 28], [9, 27], [9, 26], [9, 25], [9, 24], [9, 23], [9, 22], [9, 21], [8, 21], [7, 21], [6, 21], [5, 21], [5, 20], [5, 19], [5, 18], [5, 17], [5, 16], [5, 15], [5, 14], [5, 13], [5, 12], [5, 11], [4, 11], [4, 10], [3, 10], [3, 9], [2, 9], [1, 9], [1, 8], [1, 7], [1, 6], [1, 5], [1, 4], [1, 3], [1, 2]]
    Y1 = [[22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [10, 28], [9, 28], [9, 27], [9, 26], [9, 25], [9, 24], [9, 23], [9, 22], [9, 21], [8, 21], [7, 21], [6, 21], [5, 21], [5, 20], [5, 19], [5, 18], [5, 17], [5, 16], [5, 15], [5, 14], [5, 13], [5, 12], [5, 11], [4, 11], [4, 10], [3, 10], [3, 9], [2, 9], [1, 9], [1, 8], [1, 7], [1, 6], [1, 5], [1, 4], [1, 3], [1, 2], [2, 2]]
    X2 = []; Y2 = []
    X3 = []; Y3 = []
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)
    #2-drone:
    #Agent 1:
    agents = 2
    X1 = [[23, 28], [22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [10, 28], [9, 28], [8, 28], [8, 27], [8, 26], [8, 25], [8, 24], [8, 23], [8, 22], [8, 21], [8, 20], [8, 19], [8, 18], [8, 17], [9, 17], [10, 17], [11, 17], [12, 17], [12, 16], [12, 15], [12, 14], [12, 13], [12, 12], [12, 11], [12, 10], [12, 9], [12, 8], [12, 7], [12, 6], [11, 6], [10, 6], [9, 6], [8, 6], [7, 6], [6, 6], [5, 6], [4, 6], [4, 5], [4, 4]]
    Y1 = [[22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [10, 28], [9, 28], [8, 28], [8, 27], [8, 26], [8, 25], [8, 24], [8, 23], [8, 22], [8, 21], [8, 20], [8, 19], [8, 18], [8, 17], [9, 17], [10, 17], [11, 17], [12, 17], [12, 16], [12, 15], [12, 14], [12, 13], [12, 12], [12, 11], [12, 10], [12, 9], [12, 8], [12, 7], [12, 6], [11, 6], [10, 6], [9, 6], [8, 6], [7, 6], [6, 6], [5, 6], [4, 6], [4, 5], [4, 4], [4, 3]]
    #Agent 2:
    X2= [[25, 28], [26, 28], [27, 28], [27, 27], [27, 26], [27, 25], [27, 24], [27, 23], [27, 22], [27, 21], [27, 20], [27, 19], [27, 18], [27, 17], [27, 16], [27, 15], [27, 14], [27, 13], [27, 12], [27, 11], [27, 10], [27, 9], [27, 8], [27, 7], [27, 6], [27, 5], [27, 4], [26, 4], [25, 4], [24, 4], [23, 4], [22, 4], [21, 4], [21, 5], [21, 6], [21, 7], [20, 7], [19, 7], [18, 7], [17, 7], [16, 7], [15, 7], [14, 7], [13, 7], [12, 7], [11, 7], [10, 7], [9, 7], [8, 7], [7, 7], [6, 7], [5, 7], [4, 7], [3, 7], [3, 6], [3, 5], [3, 4]]
    Y2 = [[26, 28], [27, 28], [27, 27], [27, 26], [27, 25], [27, 24], [27, 23], [27, 22], [27, 21], [27, 20], [27, 19], [27, 18], [27, 17], [27, 16], [27, 15], [27, 14], [27, 13], [27, 12], [27, 11], [27, 10], [27, 9], [27, 8], [27, 7], [27, 6], [27, 5], [27, 4], [26, 4], [25, 4], [24, 4], [23, 4], [22, 4], [21, 4], [21, 5], [21, 6], [21, 7], [20, 7], [19, 7], [18, 7], [17, 7], [16, 7], [15, 7], [14, 7], [13, 7], [12, 7], [11, 7], [10, 7], [9, 7], [8, 7], [7, 7], [6, 7], [5, 7], [4, 7], [3, 7], [3, 6], [3, 5], [3, 4], [3, 3]]
    X3 = []; Y3 = []
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)
    #Agent 3:
    agents = 3
    X1 = [[23, 28], [22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [10, 28], [9, 28], [8, 28], [7, 28], [6, 28], [5, 28], [4, 28], [4, 27], [4, 26], [4, 25], [4, 24], [4, 23], [4, 22], [4, 21], [4, 20], [4, 19], [4, 18], [4, 17], [4, 16], [4, 15], [4, 14], [4, 13], [4, 12], [4, 11], [4, 10], [3, 10], [2, 10], [1, 10], [1, 9], [1, 8], [1, 7], [1, 6], [1, 5], [1, 4], [1, 3], [1, 2]]
    Y1 = [[22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [10, 28], [9, 28], [8, 28], [7, 28], [6, 28], [5, 28], [4, 28], [4, 27], [4, 26], [4, 25], [4, 24], [4, 23], [4, 22], [4, 21], [4, 20], [4, 19], [4, 18], [4, 17], [4, 16], [4, 15], [4, 14], [4, 13], [4, 12], [4, 11], [4, 10], [3, 10], [2, 10], [1, 10], [1, 9], [1, 8], [1, 7], [1, 6], [1, 5], [1, 4], [1, 3], [1, 2], [2, 2]]
    X2 = [[23, 28], [23, 27], [23, 26], [23, 25], [23, 24], [23, 23], [22, 23], [21, 23], [20, 23], [19, 23], [18, 23], [17, 23], [17, 22], [17, 21], [17, 20], [17, 19], [17, 18], [17, 17], [17, 16], [17, 15], [17, 14], [17, 13], [17, 12], [17, 11], [17, 10], [17, 9], [17, 8], [17, 7], [17, 6], [16, 6], [15, 6], [14, 6], [13, 6], [12, 6], [11, 6], [10, 6], [10, 5], [9, 5], [9, 4], [9, 3], [8, 3], [7, 3], [6, 3], [5, 3]]
    Y2 = [[23, 27], [23, 26], [23, 25], [23, 24], [23, 23], [22, 23], [21, 23], [20, 23], [19, 23], [18, 23], [17, 23], [17, 22], [17, 21], [17, 20], [17, 19], [17, 18], [17, 17], [17, 16], [17, 15], [17, 14], [17, 13], [17, 12], [17, 11], [17, 10], [17, 9], [17, 8], [17, 7], [17, 6], [16, 6], [15, 6], [14, 6], [13, 6], [12, 6], [11, 6], [10, 6], [10, 5], [9, 5], [9, 4], [9, 3], [8, 3], [7, 3], [6, 3], [5, 3], [4, 3]]
    X3 = [[25, 28], [26, 28], [27, 28], [28, 28], [28, 27], [28, 26], [28, 25], [28, 24], [28, 23], [28, 22], [28, 21], [28, 20], [28, 19], [28, 18], [28, 17], [28, 16], [28, 15], [28, 14], [28, 13], [28, 12], [28, 11], [28, 10], [27, 10], [26, 10], [25, 10], [24, 10], [23, 10], [22, 10], [21, 10], [20, 10], [19, 10], [18, 10], [17, 10], [16, 10], [15, 10], [14, 10], [13, 10], [12, 10], [11, 10], [10, 10], [9, 10], [8, 10], [7, 10], [6, 10], [5, 10], [4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4]]
    Y3 = [[26, 28], [27, 28], [28, 28], [28, 27], [28, 26], [28, 25], [28, 24], [28, 23], [28, 22], [28, 21], [28, 20], [28, 19], [28, 18], [28, 17], [28, 16], [28, 15], [28, 14], [28, 13], [28, 12], [28, 11], [28, 10], [27, 10], [26, 10], [25, 10], [24, 10], [23, 10], [22, 10], [21, 10], [20, 10], [19, 10], [18, 10], [17, 10], [16, 10], [15, 10], [14, 10], [13, 10], [12, 10], [11, 10], [10, 10], [9, 10], [8, 10], [7, 10], [6, 10], [5, 10], [4, 10], [4, 9], [4, 8], [4, 7], [4, 6], [4, 5], [4, 4], [4, 3]]  
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)
####### urban50
map = "urban50"
draw = 0
if draw==1:    
    #Agent 1:
    agents = 1
    X1 = [[25, 28], [25, 29], [26, 29], [26, 30], [26, 31], [27, 31], [28, 31], [29, 31], [30, 31], [31, 31], [32, 31], [33, 31], [34, 31], [35, 31], [36, 31], [37, 31], [37, 32], [37, 33], [37, 34], [37, 35], [37, 36], [37, 37], [37, 38], [37, 39], [37, 40], [37, 41], [37, 42], [37, 43], [37, 44], [37, 45], [37, 46], [37, 47], [36, 47], [35, 47], [34, 47], [33, 47], [32, 47], [31, 47], [30, 47], [29, 47], [28, 47], [27, 47], [26, 47], [25, 47], [24, 47], [23, 47], [22, 47], [21, 47], [20, 47], [19, 47], [18, 47], [17, 47], [16, 47], [15, 47], [15, 46], [15, 45], [15, 44], [15, 43], [15, 42], [15, 41], [15, 40], [15, 39], [15, 38], [15, 37], [15, 36], [15, 35], [15, 34], [16, 34], [16, 33], [17, 33], [17, 32], [18, 32], [19, 32], [20, 32], [21, 32], [22, 32], [22, 31], [22, 30], [22, 29]]
    Y1 = [[25, 29], [26, 29], [26, 30], [26, 31], [27, 31], [28, 31], [29, 31], [30, 31], [31, 31], [32, 31], [33, 31], [34, 31], [35, 31], [36, 31], [37, 31], [37, 32], [37, 33], [37, 34], [37, 35], [37, 36], [37, 37], [37, 38], [37, 39], [37, 40], [37, 41], [37, 42], [37, 43], [37, 44], [37, 45], [37, 46], [37, 47], [36, 47], [35, 47], [34, 47], [33, 47], [32, 47], [31, 47], [30, 47], [29, 47], [28, 47], [27, 47], [26, 47], [25, 47], [24, 47], [23, 47], [22, 47], [21, 47], [20, 47], [19, 47], [18, 47], [17, 47], [16, 47], [15, 47], [15, 46], [15, 45], [15, 44], [15, 43], [15, 42], [15, 41], [15, 40], [15, 39], [15, 38], [15, 37], [15, 36], [15, 35], [15, 34], [16, 34], [16, 33], [17, 33], [17, 32], [18, 32], [19, 32], [20, 32], [21, 32], [22, 32], [22, 31], [22, 30], [22, 29], [22, 28]]
    X2 = []; Y2 = []
    X3 = []; Y3 = []
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)
    #Agent 2:
    agents = 2
    X1 = [[22, 24], [21, 24], [20, 24], [19, 24], [18, 24], [17, 24], [16, 24], [15, 24], [14, 24], [13, 24], [12, 24], [11, 24], [10, 24], [9, 24], [8, 24], [7, 24], [6, 24], [6, 23], [6, 22], [6, 21], [6, 20], [6, 19], [6, 18], [7, 18], [8, 18], [9, 18], [10, 18], [10, 17], [11, 17], [12, 17], [12, 16], [12, 15], [12, 14], [12, 13], [12, 12], [12, 11], [12, 10], [13, 10], [14, 10], [15, 10], [16, 10], [17, 10], [18, 10], [18, 11], [18, 12], [18, 13], [18, 14], [18, 15], [18, 16], [18, 17], [18, 18], [18, 19], [18, 20], [18, 21], [18, 22], [19, 22], [20, 22], [21, 22]]
    Y1 = [[21, 24], [20, 24], [19, 24], [18, 24], [17, 24], [16, 24], [15, 24], [14, 24], [13, 24], [12, 24], [11, 24], [10, 24], [9, 24], [8, 24], [7, 24], [6, 24], [6, 23], [6, 22], [6, 21], [6, 20], [6, 19], [6, 18], [7, 18], [8, 18], [9, 18], [10, 18], [10, 17], [11, 17], [12, 17], [12, 16], [12, 15], [12, 14], [12, 13], [12, 12], [12, 11], [12, 10], [13, 10], [14, 10], [15, 10], [16, 10], [17, 10], [18, 10], [18, 11], [18, 12], [18, 13], [18, 14], [18, 15], [18, 16], [18, 17], [18, 18], [18, 19], [18, 20], [18, 21], [18, 22], [19, 22], [20, 22], [21, 22], [22, 22]]
    X2 = [[22, 28], [22, 29], [22, 30], [22, 31], [22, 32], [22, 33], [21, 33], [21, 34], [20, 34], [19, 34], [18, 34], [17, 34], [16, 34], [16, 35], [16, 36], [16, 37], [16, 38], [16, 39], [16, 40], [16, 41], [16, 42], [16, 43], [16, 44], [16, 45], [16, 46], [16, 47], [17, 47], [17, 46], [18, 46], [19, 46], [20, 46], [21, 46], [22, 46], [23, 46], [24, 46], [25, 46], [26, 46], [27, 46], [28, 46], [29, 46], [30, 46], [31, 46], [32, 46], [33, 46], [33, 45], [33, 44], [33, 43], [33, 42], [33, 41], [33, 40], [33, 39], [33, 38], [33, 37], [33, 36], [33, 35], [33, 34], [33, 33], [33, 32], [33, 31], [33, 30], [33, 29], [33, 28], [33, 27], [32, 27], [31, 27], [30, 27]]
    Y2 = [[22, 29], [22, 30], [22, 31], [22, 32], [22, 33], [21, 33], [21, 34], [20, 34], [19, 34], [18, 34], [17, 34], [16, 34], [16, 35], [16, 36], [16, 37], [16, 38], [16, 39], [16, 40], [16, 41], [16, 42], [16, 43], [16, 44], [16, 45], [16, 46], [16, 47], [17, 47], [17, 46], [18, 46], [19, 46], [20, 46], [21, 46], [22, 46], [23, 46], [24, 46], [25, 46], [26, 46], [27, 46], [28, 46], [29, 46], [30, 46], [31, 46], [32, 46], [33, 46], [33, 45], [33, 44], [33, 43], [33, 42], [33, 41], [33, 40], [33, 39], [33, 38], [33, 37], [33, 36], [33, 35], [33, 34], [33, 33], [33, 32], [33, 31], [33, 30], [33, 29], [33, 28], [33, 27], [32, 27], [31, 27], [30, 27], [29, 27]]
    X3 = []; Y3 = []
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)
    #Agent 3:
    agents = 3
    X1= [[22, 28], [22, 28], [22, 29], [22, 30], [22, 31], [22, 32], [22, 33], [22, 34], [22, 35], [22, 36], [22, 37], [22, 38], [22, 39], [22, 40], [22, 41], [22, 42], [22, 43], [21, 43], [20, 43], [19, 43], [18, 43], [17, 43], [16, 43], [15, 43], [14, 43], [14, 42], [14, 41], [15, 41], [16, 41], [17, 41], [18, 41], [19, 41], [20, 41], [21, 41], [21, 40], [21, 39], [21, 38], [21, 37], [21, 36], [21, 35], [21, 34], [21, 33], [21, 32], [21, 31], [21, 30], [21, 29], [21, 28], [21, 27]]
    Y1= [[22, 29], [22, 29], [22, 30], [22, 31], [22, 32], [22, 33], [22, 34], [22, 35], [22, 36], [22, 37], [22, 38], [22, 39], [22, 40], [22, 41], [22, 42], [22, 43], [21, 43], [20, 43], [19, 43], [18, 43], [17, 43], [16, 43], [15, 43], [14, 43], [14, 42], [14, 41], [15, 41], [16, 41], [17, 41], [18, 41], [19, 41], [20, 41], [21, 41], [21, 40], [21, 39], [21, 38], [21, 37], [21, 36], [21, 35], [21, 34], [21, 33], [21, 32], [21, 31], [21, 30], [21, 29], [21, 28], [21, 27], [22, 27]]
    X2= [[28, 28], [28, 28], [29, 28], [30, 28], [30, 29], [30, 30], [30, 31], [30, 32], [30, 33], [30, 34], [30, 35], [30, 36], [30, 37], [30, 38], [30, 39], [30, 40], [31, 40], [32, 40], [33, 40], [34, 40], [35, 40], [36, 40], [37, 40], [38, 40], [39, 40], [40, 40], [41, 40], [42, 40], [43, 40], [44, 40], [45, 40], [46, 40], [46, 39], [46, 38], [46, 37], [46, 36], [46, 35], [46, 34], [46, 33], [46, 32], [46, 31], [46, 30], [46, 29], [46, 28], [46, 27], [46, 26], [46, 25], [46, 24], [46, 23], [46, 22], [46, 21], [46, 20], [46, 19], [45, 19], [44, 19], [43, 19], [42, 19], [41, 19], [40, 19], [39, 19], [38, 19], [37, 19], [36, 19], [35, 19], [34, 19], [33, 19], [32, 19], [31, 19], [30, 19], [29, 19], [28, 19], [28, 20], [28, 21]]
    Y2= [[29, 28], [29, 28], [30, 28], [30, 29], [30, 30], [30, 31], [30, 32], [30, 33], [30, 34], [30, 35], [30, 36], [30, 37], [30, 38], [30, 39], [30, 40], [31, 40], [32, 40], [33, 40], [34, 40], [35, 40], [36, 40], [37, 40], [38, 40], [39, 40], [40, 40], [41, 40], [42, 40], [43, 40], [44, 40], [45, 40], [46, 40], [46, 39], [46, 38], [46, 37], [46, 36], [46, 35], [46, 34], [46, 33], [46, 32], [46, 31], [46, 30], [46, 29], [46, 28], [46, 27], [46, 26], [46, 25], [46, 24], [46, 23], [46, 22], [46, 21], [46, 20], [46, 19], [45, 19], [44, 19], [43, 19], [42, 19], [41, 19], [40, 19], [39, 19], [38, 19], [37, 19], [36, 19], [35, 19], [34, 19], [33, 19], [32, 19], [31, 19], [30, 19], [29, 19], [28, 19], [28, 20], [28, 21], [28, 22]]
    X3= [[22, 24], [22, 24], [21, 24], [20, 24], [19, 24], [18, 24], [17, 24], [16, 24], [15, 24], [14, 24], [13, 24], [12, 24], [11, 24], [10, 24], [9, 24], [8, 24], [7, 24], [6, 24], [6, 23], [6, 22], [6, 21], [6, 20], [6, 19], [7, 19], [8, 19], [9, 19], [9, 18], [10, 18], [10, 17], [11, 17], [12, 17], [12, 16], [12, 15], [12, 14], [12, 13], [12, 12], [12, 11], [12, 10], [12, 9], [13, 9], [14, 9], [15, 9], [16, 9], [17, 9], [18, 9], [19, 9], [20, 9], [20, 10], [20, 11], [20, 12], [20, 13], [20, 14], [20, 15], [20, 16], [20, 17], [20, 18], [20, 19], [20, 20], [20, 21], [20, 22], [20, 23], [21, 23]]
    Y3= [[21, 24], [21, 24], [20, 24], [19, 24], [18, 24], [17, 24], [16, 24], [15, 24], [14, 24], [13, 24], [12, 24], [11, 24], [10, 24], [9, 24], [8, 24], [7, 24], [6, 24], [6, 23], [6, 22], [6, 21], [6, 20], [6, 19], [7, 19], [8, 19], [9, 19], [9, 18], [10, 18], [10, 17], [11, 17], [12, 17], [12, 16], [12, 15], [12, 14], [12, 13], [12, 12], [12, 11], [12, 10], [12, 9], [13, 9], [14, 9], [15, 9], [16, 9], [17, 9], [18, 9], [19, 9], [20, 9], [20, 10], [20, 11], [20, 12], [20, 13], [20, 14], [20, 15], [20, 16], [20, 17], [20, 18], [20, 19], [20, 20], [20, 21], [20, 22], [20, 23], [21, 23], [22, 23]]
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)
#############################################
# Baseline
#############################################
####### manhattan32
map = "manhattan32"
model = "Baseline"
draw = 0
if draw==1:
    #Agent 1:
    agents = 1
    X1 = [[23, 28], [22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [10, 28], [9, 28], [9, 27], [9, 26], [9, 25], [9, 24], [9, 23], [9, 22], [9, 21], [8, 21], [7, 21], [6, 21], [5, 21], [5, 20], [5, 19], [5, 18], [5, 17], [5, 16], [5, 15], [5, 14], [5, 13], [5, 12], [5, 11], [5, 10], [5, 9], [5, 8], [5, 7], [5, 6], [6, 6], [7, 6]]
    Y1 = [[22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [10, 28], [9, 28], [9, 27], [9, 26], [9, 25], [9, 24], [9, 23], [9, 22], [9, 21], [8, 21], [7, 21], [6, 21], [5, 21], [5, 20], [5, 19], [5, 18], [5, 17], [5, 16], [5, 15], [5, 14], [5, 13], [5, 12], [5, 11], [5, 10], [5, 9], [5, 8], [5, 7], [5, 6], [6, 6], [7, 6], [8, 6]]
    X2 = []; Y2 = []
    X3 = []; Y3 = []
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)

    #Agent 2:
    agents = 2
    X1 = [[23, 28], [22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [10, 28], [9, 28], [9, 27], [9, 26], [9, 25], [9, 24], [9, 23], [9, 22], [9, 21], [10, 21], [11, 21], [12, 21], [13, 21], [14, 21], [15, 21], [16, 21], [17, 21], [18, 21], [19, 21], [20, 21], [21, 21], [22, 21], [23, 21], [24, 21], [25, 21], [26, 21], [27, 21], [28, 21], [28, 20], [28, 19], [28, 18], [28, 17], [28, 16]]
    Y1 = [[22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [10, 28], [9, 28], [9, 27], [9, 26], [9, 25], [9, 24], [9, 23], [9, 22], [9, 21], [10, 21], [11, 21], [12, 21], [13, 21], [14, 21], [15, 21], [16, 21], [17, 21], [18, 21], [19, 21], [20, 21], [21, 21], [22, 21], [23, 21], [24, 21], [25, 21], [26, 21], [27, 21], [28, 21], [28, 20], [28, 19], [28, 18], [28, 17], [28, 16], [28, 15]]
    X2 = [[25, 28], [26, 28], [27, 28], [28, 28], [29, 28], [29, 27], [29, 26]]
    Y2 = [[26, 28], [27, 28], [28, 28], [29, 28], [29, 27], [29, 26], [29, 25]]
    X3 = []; Y3 = []
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)

    #Agent 3:
    agents = 3
    X1 = [[23, 28], [22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [11, 27], [11, 26], [11, 25], [11, 24], [11, 23], [11, 22], [11, 21], [11, 20], [11, 19], [11, 18], [11, 17], [12, 17], [12, 16]]
    Y1 = [[22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [15, 28], [14, 28], [13, 28], [12, 28], [11, 28], [11, 27], [11, 26], [11, 25], [11, 24], [11, 23], [11, 22], [11, 21], [11, 20], [11, 19], [11, 18], [11, 17], [12, 17], [12, 16], [13, 16]]
    X2 = [[23, 28], [23, 27], [23, 26], [23, 25], [23, 24], [22, 24], [21, 24], [20, 24], [19, 24], [18, 24], [17, 24], [16, 24], [15, 24], [15, 23], [15, 22], [15, 21], [15, 20], [15, 19], [15, 18], [15, 17], [15, 16], [15, 15], [15, 14], [15, 13], [15, 12], [15, 11], [15, 10], [15, 9], [15, 8], [15, 7], [15, 6], [14, 6], [13, 6], [12, 6], [11, 6], [10, 6], [9, 6]]
    Y2 =[[23, 27], [23, 26], [23, 25], [23, 24], [22, 24], [21, 24], [20, 24], [19, 24], [18, 24], [17, 24], [16, 24], [15, 24], [15, 23], [15, 22], [15, 21], [15, 20], [15, 19], [15, 18], [15, 17], [15, 16], [15, 15], [15, 14], [15, 13], [15, 12], [15, 11], [15, 10], [15, 9], [15, 8], [15, 7], [15, 6], [14, 6], [13, 6], [12, 6], [11, 6], [10, 6], [9, 6], [8, 6]]
    X3 = [[25, 28], [26, 28], [27, 28], [28, 28], [28, 27], [28, 26], [28, 25], [28, 24], [28, 23], [28, 22], [28, 21], [28, 20], [28, 19], [28, 18], [28, 17], [28, 16], [28, 15], [28, 14], [28, 13], [28, 12], [28, 11], [28, 10], [27, 10], [26, 10], [25, 10], [24, 10], [23, 10], [22, 10], [21, 10], [20, 10], [19, 10], [18, 10], [17, 10], [17, 9], [17, 8], [17, 7], [17, 6], [17, 5], [17, 4], [17, 3], [17, 2], [17, 1], [17, 0], [16, 0], [15, 0], [14, 0], [13, 0], [12, 0], [11, 0], [10, 0], [9, 0], [8, 0], [7, 0], [6, 0], [5, 0]]
    Y3 =[[26, 28], [27, 28], [28, 28], [28, 27], [28, 26], [28, 25], [28, 24], [28, 23], [28, 22], [28, 21], [28, 20], [28, 19], [28, 18], [28, 17], [28, 16], [28, 15], [28, 14], [28, 13], [28, 12], [28, 11], [28, 10], [27, 10], [26, 10], [25, 10], [24, 10], [23, 10], [22, 10], [21, 10], [20, 10], [19, 10], [18, 10], [17, 10], [17, 9], [17, 8], [17, 7], [17, 6], [17, 5], [17, 4], [17, 3], [17, 2], [17, 1], [17, 0], [16, 0], [15, 0], [14, 0], [13, 0], [12, 0], [11, 0], [10, 0], [9, 0], [8, 0], [7, 0], [6, 0], [5, 0], [5, 1]]
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)
####### urban50
map = "urban50"
draw = 0
if draw==1:
    #Agent 1:
    agents = 1
    X1 = [[21, 22], [21, 21], [21, 20], [21, 19], [21, 18], [21, 17], [21, 16], [20, 16], [19, 16], [18, 16], [17, 16]]
    Y1 = [[21, 21], [21, 20], [21, 19], [21, 18], [21, 17], [21, 16], [20, 16], [19, 16], [18, 16], [17, 16], [16, 16]]
    X2 = []; Y2 = []
    X3 = []; Y3 = []
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)
    #Agent 2:
    agents = 2
    X1 = [[22, 22], [22, 21], [22, 20], [22, 19], [22, 18], [22, 17], [22, 16], [21, 16], [20, 16], [19, 16], [18, 16], [17, 16]]
    Y1 = [[22, 21], [22, 20], [22, 19], [22, 18], [22, 17], [22, 16], [21, 16], [20, 16], [19, 16], [18, 16], [17, 16], [16, 16]]  
    X2 = [[22, 28], [21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [16, 27], [15, 27], [15, 26]]
    Y2 = [[21, 28], [20, 28], [19, 28], [18, 28], [17, 28], [16, 28], [16, 27], [15, 27], [15, 26], [15, 25]]
    X3 = []; Y3 = []
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)
    #Agent 3:
    agents = 3
    X1 = [[22, 27], [22, 28], [22, 29], [22, 30], [22, 31], [22, 32], [22, 33], [22, 34], [22, 35], [22, 36], [22, 37], [22, 38], [22, 39], [22, 40], [22, 41], [22, 42], [22, 43], [21, 43], [20, 43], [19, 43], [18, 43], [17, 43], [16, 43], [15, 43], [14, 43], [14, 42], [14, 41], [14, 40], [14, 39], [14, 38], [14, 37], [14, 36], [14, 35], [14, 34], [14, 33], [14, 32], [14, 31], [15, 31], [15, 30], [16, 30], [16, 29], [17, 29], [17, 28], [17, 27], [18, 27], [19, 27], [20, 27], [21, 27]]
    Y1 = [[22, 28], [22, 29], [22, 30], [22, 31], [22, 32], [22, 33], [22, 34], [22, 35], [22, 36], [22, 37], [22, 38], [22, 39], [22, 40], [22, 41], [22, 42], [22, 43], [21, 43], [20, 43], [19, 43], [18, 43], [17, 43], [16, 43], [15, 43], [14, 43], [14, 42], [14, 41], [14, 40], [14, 39], [14, 38], [14, 37], [14, 36], [14, 35], [14, 34], [14, 33], [14, 32], [14, 31], [15, 31], [15, 30], [16, 30], [16, 29], [17, 29], [17, 28], [17, 27], [18, 27], [19, 27], [20, 27], [21, 27], [22, 27]]
    X2 = [[22, 25], [21, 25], [20, 25], [19, 25], [18, 25], [17, 25], [16, 25]]
    Y2 = [[21, 25], [20, 25], [19, 25], [18, 25], [17, 25], [16, 25], [15, 25]]
    X3 = [[23, 22], [23, 21], [23, 20], [23, 19], [23, 18], [22, 18], [21, 18], [20, 18], [19, 18], [18, 18], [17, 18], [16, 18], [15, 18], [14, 18], [13, 18], [13, 17], [13, 16], [13, 15], [13, 14], [13, 13], [13, 12], [13, 11], [13, 10], [14, 10], [15, 10], [16, 10], [17, 10], [18, 10], [19, 10], [20, 10], [21, 10], [22, 10], [23, 10], [24, 10], [25, 10], [26, 10], [27, 10], [28, 10], [29, 10], [30, 10], [31, 10], [32, 10], [33, 10], [34, 10], [35, 10], [36, 10], [37, 10], [38, 10], [38, 11], [38, 12], [38, 13], [38, 14], [38, 15], [38, 16], [38, 17], [38, 18], [39, 18], [40, 18], [41, 18], [42, 18], [43, 18], [43, 19], [42, 19], [41, 19], [40, 19], [39, 19], [38, 19], [37, 19], [36, 19], [35, 19], [34, 19], [33, 19], [32, 19], [31, 19], [30, 19], [29, 19], [28, 19], [28, 20], [28, 21]]
    Y3 = [[23, 21], [23, 20], [23, 19], [23, 18], [22, 18], [21, 18], [20, 18], [19, 18], [18, 18], [17, 18], [16, 18], [15, 18], [14, 18], [13, 18], [13, 17], [13, 16], [13, 15], [13, 14], [13, 13], [13, 12], [13, 11], [13, 10], [14, 10], [15, 10], [16, 10], [17, 10], [18, 10], [19, 10], [20, 10], [21, 10], [22, 10], [23, 10], [24, 10], [25, 10], [26, 10], [27, 10], [28, 10], [29, 10], [30, 10], [31, 10], [32, 10], [33, 10], [34, 10], [35, 10], [36, 10], [37, 10], [38, 10], [38, 11], [38, 12], [38, 13], [38, 14], [38, 15], [38, 16], [38, 17], [38, 18], [39, 18], [40, 18], [41, 18], [42, 18], [43, 18], [43, 19], [42, 19], [41, 19], [40, 19], [39, 19], [38, 19], [37, 19], [36, 19], [35, 19], [34, 19], [33, 19], [32, 19], [31, 19], [30, 19], [29, 19], [28, 19], [28, 20], [28, 21], [28, 22]]
    RealTimePlotting(X1,Y1,X2,Y2,X3,Y3, map, model,agents)

print("")