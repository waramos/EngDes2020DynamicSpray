#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 11:37:20 2019

@author: alexcohen
"""

# Import necessary packages
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math


# Defining a class to hold the triangle and normal vector data in .stl
class Segment:
    #Attributes
    def __init__(self, normal, v1, v2, v3):
        self.normal = normal
        self.v1 = v1
        self.v2 = v2
        self.v3 = v3
        
    def output(self):
        return "Normal = {},{},{}\nV1 = {},{},{}\nV2 = {},{},{}\nV3 = {},{},{}".format(self.normal[0], self.normal[1], self.normal[2], self.v1[0], self.v1[1], self.v1[2], self.v2[0], self.v2[1], self.v2[2], self.v3[0], self.v3[1], self.v3[2])
    
# Function for reading in an stl ascii file
# INPUTS : fname --> name of stl file
# OUTPUTS: res --> array of Segments
#          i --> number of lines in file
def intake(fname):
    res = []
    fhandle = open(fname)
    line = fhandle.readline()
    i = 0
    while(line != ''):# and i < 10):
        splitline = line.split()
        tempNorm = []
        tempV = [[],[],[]]
        if splitline[0] == "facet":
            tempNorm = np.array(splitline[2:], dtype="float")
            line = fhandle.readline()
            for j in range(3):
                splitline = fhandle.readline().split()
                tempV[j] = np.array(splitline[1:], dtype="float")
            res.append(Segment(tempNorm, tempV[0], tempV[1], tempV[2]))
        i += 1
        line=fhandle.readline()
    return res,i

# Function for finding the intersection between a plane and line
# INPUTS:
#   pN --> plane normal vector
#   pP --> point on plane
#   rD --> direction of ray
#   rP --> point on ray
#   epsilon --> tolerance for intersection
# OUTPUTS
#   Psi --> intersection
def intersection(pN, pP, rD, rP, epsilon = .00001):
    ndotu = np.dot(pN,rD) 
    
    if abs(ndotu) < epsilon:
        return [-10000,0,0]

    w = rP - pP
    si = -pN.dot(w) / ndotu
    Psi = w + si * rD + pP
    return Psi

########## Read in an .stl file ##########
test,lines = intake("sphere.stl")
#test, lines = intake("Sphericon.stl")
#test, lines = intake("sphere.stl")
#test, lines = intake("liver.stl")
####################

########## Organize data from stl file into usable format ##########
x = []
y = []
z = []
xs = []
ys = []
zs = []
xs1 = []
ys1 = []
zs1 = []

for i in range(0,len(test), 100):
    #print(test[i].output())
    tempx = []
    tempy = []
    tempz = []
    tempx.append(test[i].v1[0])
    tempx.append(test[i].v2[0])
    tempx.append(test[i].v3[0])
    tempy.append(test[i].v1[1])
    tempy.append(test[i].v2[1])
    tempy.append(test[i].v3[1])
    tempz.append(test[i].v1[2])
    tempz.append(test[i].v2[2])
    tempz.append(test[i].v3[2])
    x.append(test[i].v1[0])
    x.append(test[i].v2[0])
    x.append(test[i].v3[0])
    y.append(test[i].v1[1])
    y.append(test[i].v2[1])
    y.append(test[i].v3[1])
    z.append(test[i].v1[2])
    z.append(test[i].v2[2])
    z.append(test[i].v3[2])
    xs.append(tempx)
    ys.append(tempy)
    zs.append(tempz)

    xs1.append(test[i].normal[0])
    ys1.append(test[i].normal[1])
    zs1.append(test[i].normal[2])

####################

########## Plot a 3D image of the stl file ##########
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.scatter(x,y,z)

for i in test:
    ax.plot([i.v1[0],i.v2[0],i.v3[0],i.v1[0]], [i.v1[1],i.v2[1],i.v3[1],i.v1[1]], [i.v1[2],i.v2[2],i.v3[2],i.v1[2]])
    
####################


# Check if a vector is normalized
def normalized(vec, eps):
    if np.sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]) - 1 < eps:
        return True
    return False

# Returns the projection of a point onto a plane
def point_proj(point, normal, vertex):
    return vertex - (np.dot(normal, vertex-point))*normal

# Returns the center of a list of points
def get_center(points):
    num_points = len(points)
    com = np.array([0,0,0])
    for p in points:
        com[0] += p[0]
        com[1] += p[1]
        com[2] += p[2]
    com[0] /= num_points
    com[1] /= num_points
    com[2] /= num_points
    return com

# Calculates distance between two points
def distance(p1,p2):
    return np.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2 + (p2[2]-p1[2])**2)

# Returns the maximum range of all vertices projected onto a plane
# Used to determine the bounds of the object
def shape_range(point, normal, stl):
    temp = []
    temp_origin = []
    for tri in stl:
        ppv1 = point_proj(point, normal, tri.v1)
        ppv2 = point_proj(point, normal, tri.v2)
        ppv3 = point_proj(point, normal, tri.v3)
        origv1 = point_proj(np.array([0,0,0]), np.array([0,0,1]), tri.v1)
        origv2 = point_proj(np.array([0,0,0]), np.array([0,0,1]), tri.v2)
        origv3 = point_proj(np.array([0,0,0]), np.array([0,0,1]), tri.v3)
        temp.append(ppv1)
        temp.append(ppv2)
        temp.append(ppv3)
        temp_origin.append(origv1)
        temp_origin.append(origv2)
        temp_origin.append(origv3)
    com = get_center(temp_origin)
    dist = 0
    for p in temp_origin:
        d = distance(com, p)
        if d > dist:
            dist = d
    diag = dist*np.sqrt(2) + 1
    corners = [[com[0]+diag,com[1],0], [com[0]-diag,com[1],0], [com[0], com[1]+diag, 0], [com[0], com[1]-diag, 0]]
    res = []
    for c in corners:
        res.append(point_proj(point, normal, c))
    return res
        
########## DEFINE THE point and orientation direction to find the thickness profile ##########
planeNormal = np.array([0,0,1])
planePoint = np.array([50, 50, 0])
####################

#ax.set_zlim3d(0, 100)                    
#ax.set_ylim3d(0,100)                   
#ax.set_xlim3d(0,100)


# Finds the points to include in the projection
def included_points(stl, point, normal, resolution):
    corners = shape_range(point, normal, stl)
    xycorn = []
    for c in corners:
        xycorn.append(point_proj(point, normal, c))
    maxX = xycorn[0][0]
    minX = xycorn[0][0]
    maxY = xycorn[0][1]
    minY = xycorn[0][1]
    for c in xycorn:
        if c[0] > maxX:
            maxX = c[0]
        if c[0] < minX:
            minX = c[0]
        if c[1] > maxY:
            maxY = c[1]
        if c[1] < minY:
            minY = c[1]
    xs = np.linspace(minX,maxX,resolution)
    ys = np.linspace(minY,maxY,resolution)
    realpoints = []
    for x in xs:
        for y in ys:
            realpoints.append(point_proj(point, normal, [x,y,0]))
            #ax.scatter(realpoints[-1][0], realpoints[-1][1], realpoints[-1][2])
    return realpoints


# Returns the normalized vector corresponding to the given vector
def norm(v):
    n = distance(v,[0,0,0])
    #print(n)
    return np.array(v)/n

# Returns true if the given point is in the given triangle
def in_triangle(v1,v2,v3,p, eps):
    anglesum = 0
    norm1 = norm(v1-p)
    norm2 = norm(v2-p)
    norm3 = norm(v3-p)
    anglesum += np.arccos(np.dot(norm1,norm2))
    anglesum += np.arccos(np.dot(norm1,norm3))
    anglesum += np.arccos(np.dot(norm2,norm3))
    #print(anglesum)
    if abs(anglesum - 2*math.pi) < eps:
        return True
    return False
    

# Returns the thickness profile of a shape given the stl information and the direction of perspective
def thickness_profile(stl, point, nrml, resolution):
    points = included_points(stl, point, nrml, resolution)
    #points = []
    '''
    for i in range(5 + 1):
        x = ((i - 5/2) * 60e-6/5)/10e-7 * 1.99 + 55       # maps index i to coordinate x
        for j in range(5 + 1):
            y = ((j - 5/2) * 60e-6/5)/10e-7 * 1.99 + 55
            points.append([x,y,0])
    '''
    thickness = []
    hits = []
    for p in points:
        intersections = []
        #ax.scatter(p[0],p[1],p[2])
        for tri in stl:
            inter = intersection(norm(tri.normal), tri.v1, nrml, p, 0.00001)
            #print(inter)
            if inter[0] != -10000:
                if in_triangle(np.array(tri.v1), np.array(tri.v2), np.array(tri.v3), np.array(inter), 0.01):
                    intersections.append(inter)
                    #ax.scatter(p[0],p[1],p[2])
        #print(len(intersections))
        if (len(intersections) == 2):
            thickness.append(distance(intersections[0], intersections[1]))
            hits.append(p)
        else:
            thickness.append(0)
            hits.append(p)
    return thickness, hits

# Finds thickness profile with corresponding perspective and resolution
thick, hts=thickness_profile(test, planePoint, planeNormal, 50)
xs = []
ys = []
for i in range(len(hts)):
    #xs.append((hts[i][0]-55)/1.99* 10e-7)
    #ys.append((hts[i][1] - 55)/1.99 * 10e-7)
    xs.append(hts[i][0])
    ys.append(hts[i][1])
    #ax.scatter(hts[i][0], hts[i][1], c=thick[i],cmap="RdPu")
for i in range(1,len(hts)-1):
    if thick[i-1] > 0 and thick[i+1] > 0 and thick[i] == 0:
        thick[i] = (thick[i-1] + thick[i+1])/2
   
    
########## Print the thickness profile ##########
#print(xs)
#print(ys)
fig1 = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(xs,ys,c=thick,cmap="hot")
#cbar = fig1.colorbar()
#cbar.set_label("Thickness")

####################

#ax1.set_xlabel("X")
#ax1.set_ylabel("Y")
#ax.set_xlim((-16*10e-6,16*10e-6))
#ax.set_ylim((-16*10e-6,16*10e-6))
#ax.set_zlabel("Z")

        
#ax.plot3D()