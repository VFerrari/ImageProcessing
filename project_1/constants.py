# Constants for project 1
from numpy import array

FLOYD = 1/16*array([[0,0,7],
                    [3,5,1]])

STEVENSON = 1/200*array([[0 , 0, 0, 0, 0,32, 0],
                         [12, 0,26, 0,30, 0,16],
                         [0 ,12, 0,26, 0,12, 0],
                         [5 , 0,12, 0,12, 0, 5]])

BURKES = 1/32*array([[0,0,0,8,4],
                     [2,4,8,4,2]])

SIERRA = 1/32*array([[0,0,0,5,3],
                     [2,4,5,4,2],
                     [0,2,3,2,0]])

STUCKI = 1/42*array([[0,0,0,8,4],
                     [2,4,8,4,2],
                     [1,2,4,2,1]])

JARVIS = 1/48*array([[0,0,0,7,5],
                     [3,5,7,5,3],
                     [1,3,5,3,1]])

DISTS = [FLOYD,STEVENSON,BURKES,SIERRA,STUCKI,JARVIS]
NAMES = ['FLOYD','STEVENSON','BURKES','SIERRA','STUCKI','JARVIS']

def select_dist(name):
    dist = DISTS if name.lower()=='all' else [DISTS[NAMES.index(name.upper())]]
    return dist
