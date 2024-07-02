from dataclasses import dataclass
import time
from typing import Optional
import math
import Box2D
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
import pandas as pd
import numpy as np

# Set up the Box2D world
# world = Box2D.b2World(gravity=(0,0))

@dataclass
class Circle():
    x:float
    y:float
    r:float
    id:str
    parent:Optional['Circle']=None
    size:Optional[float]=None

# Define a function to create a circle body
def create_circle(world, x, y, radius, density=1.0):
    body_def = Box2D.b2BodyDef()
    body_def.type = Box2D.b2_dynamicBody
    body_def.position = (x , y )
    body = world.CreateBody(body_def)

    shape = Box2D.b2CircleShape(radius=radius)
    fixture_def = Box2D.b2FixtureDef(shape=shape, density=density)
    body.CreateFixture(fixture_def)

    return body

def GetRadii(sizes, min_size=0, scale=1, force=None):
    # expect a series of cluster sizes
    if force is not None:
        sel=sizes[(sizes>=min_size)|(force)]
    else:
        sel=sizes[sizes>=min_size]
    radii=[]
    for s in sel:
        radii.append((s*scale/math.pi)**0.5)
    radii=pd.Series(radii, index=sel.index, name='radius')
    return radii

def MakeLayout(radii, radius_pad=1, space_mult=1.2, cycles=10_000, time_step=1, velocity_iterations=6, position_iterations=2,
                parent:Circle=None, seed=None, pull=0, min_radius_frac=.05, seed_coords=None, expand_mult=1, metadata=None, silence=True):
    '''
    radii: pandas Series of radii (index indicates the id of the circle)
    parent: parent circle to nest current circles within
    seed: random seed
    seed_coords: Series of (x,y) coordinates to use as starting points for circles, indexed by the id which is expected to match radii
    '''
    if seed is not None:
        random.seed(seed)

    # if parent is None:
    area=sum(math.pi*(r+radius_pad)**2 for r in radii)
    spacing=area**0.5 * space_mult

    world = Box2D.b2World(gravity=(0,0))
    # bodies=[]
    rmin, rmax = min(radii), max(radii)
    min_effective_r = rmax * min_radius_frac
    total_area=0
    start_circles=[]
    N=len(radii)
    lastx, lasty, lastr=0, 0, 0
    distances=[]
    radii_sum=[]
    for i, r in radii.items():
        if seed_coords is not None:
            row=seed_coords.loc[i] # match by index
            x, y=row['x'], row['y']
        else:
            angle=random.random() * 2 * math.pi
            pos_r=2**-math.log(max(min_effective_r, r)) * spacing + random.random() * spacing * .01
            x=pos_r*math.cos(angle)
            y=pos_r*math.sin(angle)
        total_area+=math.pi*(r+radius_pad)**2
        # bodies.append(create_circle(world, x, y, r+radius_pad, r**2))
        start_circles.append(Circle(x, y, r+radius_pad, i))
        # track distances
        if i != radii.index[0]:
            distances.append(((x-lastx)**2+(y-lasty)**2)**0.5)
            radii_sum.append(r+lastr)
        lastx, lasty, lastr=x, y, r
    if len(start_circles)>1:
        avg_dist=sum(distances)/len(distances)
        avg_radii_overlap=sum([radii_sum[i]-distances[i] for i in range(len(radii_sum))])/len(radii_sum)
    else:
        avg_dist=0
        avg_radii_overlap=0

    # get boundaries
    xmax, xmin, ymax, ymin=GetLimits(start_circles)
    # space the circles out so that the bounding box is 1.2 times the area of the circles
    if seed_coords is None:
        expand_factor=1
        pull_mod=1
    else:
        expand_factor=(total_area/(xmax-xmin)/(ymax-ymin)) * expand_mult * len(radii)**.5 * avg_radii_overlap
        pull_mod=expand_factor/cycles
    
    # get centroid of xy
    # if expand_mult!=1:
    cx, cy=0, 0
    for c in start_circles:
        cx+=c.x
        cy+=c.y
    cx/=len(start_circles)
    cy/=len(start_circles)

    start=time.time()
    if N >100 and not silence:
        print(f' Average distance: {avg_dist}, average overlap: {avg_radii_overlap}')
        print(f' Expanding by {expand_factor} for {len(radii)} circles')

    # scale distance from centroid
    for c in start_circles:
        c.x=(c.x-cx)*expand_factor+cx
        c.y=(c.y-cy)*expand_factor+cy

    if N >100 and not silence:
        print(f'  Expanding took {time.time()-start} seconds')

    bodies=[]
    for c in start_circles:
        bodies.append(create_circle(world, c.x, c.y, c.r, c.r**2))

    for cycle in range(cycles):
        cycle_mod=2 if cycle<cycles*2/3 else 1
        if N>1000:
            cycle_start=time.time()
        if pull!=0:
            for c in bodies:
                x,y=c.position
                mod=c.fixtures[0].density * pull * pull_mod
                c.ApplyForceToCenter((-x*mod,-y*mod), True)
        world.Step(time_step*cycle_mod, velocity_iterations, position_iterations)
        if N>1000 and not silence:
            print(f'  Cycle took {time.time()-cycle_start} seconds')

    if N>100 and not silence:
        print(f' Layout took {time.time()-start} seconds')
        

    circles=[] 
    for i in range(len(bodies)):
        x,y=bodies[i].position
        id=radii.index[i]
        circles.append(Circle(x, y, radii[id], id, parent))
    xmax, xmin, ymax, ymin=GetLimits(circles)

    # get center to off set grouping
    cx=(xmax+xmin)/2    
    cy=(ymax+ymin)/2

    xoff, yoff= 0, 0
    if parent is not None:
        xoff, yoff=parent.x, parent.y
    for c in circles:
        c.x=c.x-cx+xoff
        c.y=c.y-cy+yoff

    return circles, metadata

def GetLimits(circles, pad=0):
    xmax, xmin, ymax, ymin=circles[0].x, circles[0].x, circles[0].y, circles[0].y
    for c in circles:
        xmax=max(xmax, c.x+c.r)
        xmin=min(xmin, c.x-c.r)
        ymax=max(ymax, c.y+c.r)
        ymin=min(ymin, c.y-c.r)
    return xmin-pad, xmax+pad, ymin-pad, ymax+pad

if __name__=='__main__':

    # test code
    n=10_000
    radii=GetRadii(pd.Series([random.random()*100+1 for x in range(n)]))
    circles=MakeLayout(radii)
    xmin, xmax, ymin, ymax=GetLimits(circles)

    f, ax=plt.subplots(figsize=(10,10))
    ax.axis('off')

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    for c in circles:
        ax.add_patch(plt.Circle((c.x, c.y), c.r, alpha=.2, linewidth=1, fill=False))
    plt.show()

