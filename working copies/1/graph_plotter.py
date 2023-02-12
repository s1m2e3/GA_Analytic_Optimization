#Kyle Norland
#1/24/23
#Custom visualizer for grid world Markov Chains
#Also for visualizing Q-Learning values.

#-------------
#--Imports----
#-------------
import pygame
import sys
from pygame.locals import *
import time
import math

#-------------------
#-------Functions----
#--------------------
WHITE=(255,255,255)
BLACK=(0,0,0)
BLUE=(0,0,255)
GREEN=(0,255,0)
RED=(255,0,0)
YELLOW=(255,255,0)

def draw_grid(screen, x, y, width, height, box_width, box_height, border_width):
    #x,y is upper left corner of overall grid
    #middle points stores where the middle of each square is.
    
    middle_points = []
    for i in range(height):
        print(i)
        points_row =[] 
        for j in range(width):
            pygame.draw.rect(screen, BLACK, (x + j*box_width, y + i*box_height, box_width, box_height), width=border_width)
            #Save the middle point
            points_row.append((x + j*box_width + (0.5*box_width), y + i*box_height + (0.5*box_height)))
        
        middle_points.append(points_row)    
    #Return middle points of array.
    return middle_points

def angle(A, B, aspectRatio):
    x = B[0] - A[0]
    y = B[1] - A[1]
    angle = math.atan2(-y, x / aspectRatio)
    return angle
    
def angle_two(A,B):
    x = B[0] - A[0]
    y = B[1] - A[1]
    angle = math.atan2(-y, x)
    return angle
    
def distance(A, B):
        return math.hypot(B[0] - A[0], B[1] - A[1])    
    
def plot_edges(screen, nrows, ncolumns, midpoints, edges, color=(0,0,0)):
    #Plots a slightly curved path from start to finish
    global aspect_ratio
    
    for entry in edges:
        #print(entry)
        #Get first point
        first_index = to_2d_index(nrows, ncolumns, entry[0])
        first_center = midpoints[first_index[0]][first_index[1]]
        
        #Get second point
        second_index = to_2d_index(nrows, ncolumns, entry[1])
        second_center = midpoints[second_index[0]][second_index[1]]
        
        #Calculate distance
        interpoint_distance = distance(first_center, second_center)
        
        #Figure out which is left and right (should also be top if on same plane. As well as a flag for which way to put the center.
        center_flag = 'left'
        if (first_center[0] < second_center[0]) or (first_center[0] == second_center[0] and first_center[1] > second_center[1]):
            left_point = first_center
            right_point = second_center
            center_flag = 'left'
            #print('a')
            
        elif (first_center[0] == second_center[0] and first_center[1] > second_center[1]):
            left_point = first_center
            right_point = second_center
            center_flag = 'left'
            #print('b')
 
        else:
            left_point = second_center
            right_point = first_center
            center_flag = 'left'
            #print('c')
            
        #print("Left_point", left_point)
        #print("Right_point", right_point)
        #print("Center_flag", center_flag)
            
        #------Choose radius---------------
        r = distance(left_point, right_point)
        
           
        #Calculate the midpoint and angle
        #straight_angle = angle(left_point, right_point, aspect_ratio)
        
        straight_angle = angle_two(left_point, right_point)
        
        points_center = (left_point[0] + (0.5*(right_point[0]-left_point[0])), (left_point[1] + (0.5*(right_point[1]-left_point[1]))))
        #print("points_center", points_center)
        
        #perpendicular angle (to left)
        if center_flag == 'left':
            perp_angle = straight_angle + (math.pi/2)
        if center_flag == 'right':
            perp_angle = straight_angle - (math.pi/2)
        
        #print("straight_angle", straight_angle)
        #print("perpendicular_angle", perp_angle)
        #Calculate center point
        a = r
        b = distance(left_point, right_point) 
        h = math.sqrt(a**2 - (0.5*b)**2)
        #print("h", h)
        
        center_point = (points_center[0] + h*math.cos(perp_angle), points_center[1] - h*math.sin(perp_angle))
        #print("x offset", h*math.cos(perp_angle))
        #print("y offset", h*math.sin(perp_angle))
        #print('center_point', center_point)
        
        #Plot both
        #pygame.draw.circle(screen, RED, points_center, radius=5, width=0)
        #pygame.draw.circle(screen, RED, center_point, radius=5, width=0)
        
        #pts = [left_point, right_point]
        #distances = [distance(center_point, p) for p in pts]
        #angles = [angle(center_point, p, aspect_ratio) for p in pts]
        
        #pygame.draw.arc(screen, (0, 0, 0), (center[0] - lenA // 2, center[1] - lenB // 2, lenA, lenB), *angles)

        
        #pygame.draw.arc(screen, (0, 0, 0), (center[0] - lenA // 2, center[1] - lenB // 2, lenA, lenB), *angles)
        #calculate angles
        
        #start_angle = (math.pi)/math.pi*math.atan2(left_point[1]-center_point[1], left_point[0]-center_point[0])
        #end_angle = (math.pi)/math.pi*math.atan2(right_point[1]-center_point[1], right_point[0]-center_point[0])
        
        #if center_flag == 'left':
        #start_angle = (math.pi)/math.pi*math.atan2(right_point[1]-center_point[1], right_point[0]-center_point[0])
        #end_angle = (math.pi)/math.pi*math.atan2(left_point[1]-center_point[1], left_point[0]-center_point[0])
        
        start_angle = angle_two(center_point, left_point) + (math.pi/2)
        end_angle = angle_two(center_point, right_point) + (math.pi/2)
        
        if center_flag == 'right':    
            end_angle = end_angle + math.pi
            start_angle = start_angle + math.pi
        
        x = int(center_point[0] - r)
        y = int(center_point[1] - r)
        width = 2*r
        height = 2*r
        
        #print("Points", right_point, left_point)
        #print(start_angle, end_angle, x,y,width, height)
        start_degrees = math.degrees(start_angle)
        end_degrees = math.degrees(end_angle)
        if start_degrees < 0: start_degrees += 360
        if end_degrees < 0: end_degrees += 360
        #print(start_degrees, end_degrees)
        
        #pygame.draw.arc(screen, (0, 0, 0), (x, y, width, height), *angles)
        
        start_angle = start_angle - math.pi/2
        end_angle = end_angle - math.pi/2
        
        #Draw the arc
        pygame.draw.arc(screen, color, (x, y, width, height), start_angle, end_angle, width=1)
        
        #Draw the triangle at the end of the arc
        triangle_point = (0,0)
        triangle_angle = 0
        if second_center == right_point:
            triangle_point = second_center
            triangle_angle = end_angle + ((1/2)*math.pi)
            #print("Case a")
        elif second_center == left_point:
            triangle_point = second_center
            triangle_angle = start_angle + ((1/2)*math.pi) + math.pi
            #print("Case b")
        
        #print("triangle_point", triangle_point)
        #print("triangle_angle", triangle_angle)
        
        triangle_points = []
        #3 Times find the points at 120 degrees further clockwise. 2/3 radians.
        side_length = 10
        cp = triangle_point
        ca = triangle_angle
        #angle_series = [(1/6)*math.pi, (2/3)*math.pi, (1/6)*math.pi
        #Draw a check line for direction
        pygame.draw.line(screen,(0,0,0), cp, (cp[0] + side_length*math.cos(ca), cp[1] - side_length*math.sin(ca)))
        
        #First one
        next_angle = ca + (math.pi - ((1/6) * math.pi)) #Plus 150 degrees for 30 degree angle from straight up.
        next_x = cp[0] + side_length*math.cos(next_angle)
        next_y = cp[1] - side_length*math.sin(next_angle)
        triangle_points.append((next_x, next_y))
        cp = (next_x, next_y)
        ca = (next_angle)
        
        #Second
        next_angle = ca + ((2/3) * math.pi) #Plus 120 degrees for 60 degree angle from straight up.
        next_x = cp[0] + side_length*math.cos(next_angle)
        next_y = cp[1] - side_length*math.sin(next_angle)
        triangle_points.append((next_x, next_y))
        cp = (next_x, next_y)
        ca = (next_angle)        
        
        #Third
        next_angle = ca + ((2/3) * math.pi) #Plus 120 degrees for 60 degree angle from straight up.
        next_x = cp[0] + side_length*math.cos(next_angle)
        next_y = cp[1] - side_length*math.sin(next_angle)
        triangle_points.append((next_x, next_y))
        cp = (next_x, next_y)
        ca = (next_angle)  
        
        #Draw polygon of triangle and fill with same color as line.
        #print("triangle_points: ", triangle_points)
        pygame.draw.polygon(screen, color, triangle_points)
        
        
        #Break out of loop
        #break
        
        
        
        '''
        
        #Calculate the start and finish angles?
        #Use sin/cos
        #radius = distance // 10 
        
        #Calculate x,y, width, height
        x = min(first_center[0], second_center[0])
        y = min(first_center[1], second_center[1])
        width = max(first_center[0], second_center[0]) - x
        height = max(first_center[1], second_center[1]) - y   
        
        pi = 3.14159
        print(x, y, width, height)
        pygame.draw.arc(screen, GREEN, [x, y, width, height], pi, pi/3, width=10)
        pygame.draw.line(screen, GREEN, (100,100), (200,200))
        


        a1 = angle((250, 175), (75, 300), 500/350)
        a2 = angle((250, 175),(400, 315), 500/350)

        pygame.draw.arc(gameDisplay, WHITE, (0, 0, 500, 350), a1, a2)
        '''
        
def plot_known_rewards(screen, nrows, ncolumns, midpoints, known_rewards):
    for entry in known_rewards:
        #Convert to 2d index
        index = to_2d_index(nrows, ncolumns, entry)
        center = midpoints[index[0]][index[1]]
        
        #Details
        box_width = 30
        box_height = 30
        
        #Plot rectangle at midpoint
        pygame.draw.rect(screen, YELLOW, (int((center[0] - (0.5* box_width))), int((center[1] - (0.5 * box_height))), box_width, box_height), width=0)       

def plot_explored(screen, nrows, ncolumns, midpoints, explored):
    for entry in explored:
        #Convert to 2d index
        index = to_2d_index(nrows, ncolumns, entry)
        center = midpoints[index[0]][index[1]]
        
        #Plot circle at midpoint
        pygame.draw.circle(screen, GREEN, center, radius=10, width=0)

def to_2d_index(nrows, ncolumns, index):
    #print("index: ", index)
    row = index // ncolumns
    column = index % ncolumns
    return (row, column)

def initialize_plot():
    #Set up pygame
    pygame.init()# Setting your screen size with a tuple of the screen width and screen height
    screen_width = 800
    screen_height = 600
    aspect_ratio = screen_width/screen_height
    screen = pygame.display.set_mode((800,600))# Setting a random caption title for your pygame graphical window.
    pygame.display.set_caption("pygame test")
    
    '''
    #Blank out screen
    screen.fill(WHITE)
    
    #Graph information
    nrows = 4
    ncolumns = 4
    
    #Initial Draw
    midpoints = draw_grid(50, 50, ncolumns, nrows, 76, 76, 2)
    #print("Middle points", midpoints)
    
    
    #Array of explored states (1-16, etc)
    #position = current_row*nrows + current_col

    explored = [0,1,5,8,10]
    known_rewards = [0,5,9,12]
    known_edges = [(8,9),(10,5), (9,6),  (5,9), (2,12)]
    '''
    return screen

   
def draw_plot(screen, nrows, ncolumns, explored, known_rewards, known_edges, greedy_path=[], optim_path=[], q_edges=[(0,2)]):
    #Blank out screen
    screen.fill(WHITE)
    
    #Graph information
    #nrows = 4
    #ncolumns = 4
    
    #Initial Draw
    midpoints = draw_grid(screen, 50, 50, ncolumns, nrows, 76, 76, 2)
    #print("Middle points", midpoints)
    
    plot_known_rewards(screen, nrows, ncolumns, midpoints, known_rewards)
    plot_explored(screen, nrows, ncolumns, midpoints, explored)
    #Plot known edges
    plot_edges(screen, nrows, ncolumns, midpoints, known_edges, color=(119,255,51))
    
    #Plot optim path
    plot_edges(screen, nrows, ncolumns, midpoints, optim_path, color=(255,0,0))
    
    #Plot greedy path
    plot_edges(screen, nrows, ncolumns, midpoints, greedy_path, color=(255,0,255))
    

    
    #Plot q_edges
    #plot_edges(screen, nrows, ncolumns, midpoints, q_edges, color=(255,200,0))
    pygame.display.update()

#----------------------------
#---------Main---------------
#----------------------------
if __name__ == "__main__":
    #Set up pygame
    pygame.init()# Setting your screen size with a tuple of the screen width and screen height
    screen_width = 800
    screen_height = 600
    aspect_ratio = screen_width/screen_height
    screen = pygame.display.set_mode((800,600))# Setting a random caption title for your pygame graphical window.
    pygame.display.set_caption("pygame test")
    
    #Blank out screen
    screen.fill(WHITE)
    
    #Graph information
    nrows = 4
    ncolumns = 4
    
    #Initial Draw
    midpoints = draw_grid(50, 50, ncolumns, nrows, 76, 76, 2)
    print("Middle points", midpoints)
    
    
    #Array of explored states (1-16, etc)
    #position = current_row*nrows + current_col

    explored = [0,1,5,8,10]
    known_rewards = [0,5,9,12]
    known_edges = [(8,9),(10,5), (9,6),  (5,9), (2,12)]
    
    #Plot known data
    
    plot_known_rewards(nrows, ncolumns, midpoints, known_rewards)
    plot_explored(nrows, ncolumns, midpoints, explored)
    plot_known_edges(nrows, ncolumns, midpoints, known_edges)
    
    #Plot edges
    
    #Plot solved edges
    
    
    
    
    
    #Run Loop
    while True:
        #Check for exit
        for event in pygame.event.get():
            if event.type==QUIT:
                pygame.quit()
                sys.exit()
        #Update screen        
        pygame.display.update()  
        time.sleep(0.25)







