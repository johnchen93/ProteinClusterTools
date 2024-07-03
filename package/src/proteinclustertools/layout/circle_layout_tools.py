
import math
from multiprocessing import Pool
import os
from types import SimpleNamespace
from .circle_collision import Circle, GetRadii, MakeLayout, GetLimits
import pandas as pd
import yaml

def ReadConfig(config_file):
    config_dict=yaml.safe_load(open(config_file))
    config=SimpleNamespace(**config_dict)
    return config

def CommentedConfigString(config:SimpleNamespace):
    cstr=yaml.safe_dump(config.__dict__)
    commented='#'+cstr.replace('\n','\n#')
    return commented

def ConfigFromComments(text):
    cstr=text[1:].replace('\n#','\n')
    config=SimpleNamespace(**yaml.safe_load(cstr))
    return config

def process_child_clusters(args):
    '''Process child clusters in parallel
        Has same inputs as MakeLayout, but with child_clusters instead of radii as the first argument
    '''
    child_clusters, radii, radius_pad, space_scale, seed, seed_pos, cycles, parent, pull, expand_mult, velocity_iterations, position_iterations, time_step, metadata = args
    return MakeLayout(radii.loc[child_clusters], radius_pad=0, space_mult=space_scale*.8, seed=seed, seed_coords=seed_pos,
                      time_step=time_step, cycles=cycles, parent=parent, pull=pull, expand_mult=expand_mult,
                      velocity_iterations=velocity_iterations, position_iterations=position_iterations, metadata=metadata)

def CenterOfMass(circles): # not used
    x, y=0, 0
    for c in circles:
        x+=c.x
        y+=c.y
    x/=len(circles)
    y/=len(circles)
    return x, y

def Box2DLayout(clusters, levels, minsize=0, space_scale=1.1, pull=1, radius_pad=3, force_inclusion=None, velocity_iterations=16, position_iterations=6,
                cycle_scale=1000, max_cycles=100_000, time_step=1, pull_base=False, seed=None, seed_coords=None, expand_mult=1):
    
    '''
    Function to turn a table of cluster definitions into a hierarchical layout of circles

    Args:
        General:
            - clusters (pd.DataFrame): table of cluster definitions, with columns 'id' for each sequence one column per "level" of clustering
            - levels (list): list of column names in clusters to use as levels of clustering. Must be in ascending order (lowest cut-off first) of hierarchy
            - minsize (int): minimum size of a cluster to be included in the layout
            - force_inclusion (list): list of sequence ids to force inclusion in the layout
        Tuning layout:
            - pull (float): multiplier for the pull between circles when packing
            - radius_pad (float): padding between circles in the base (lowest) level
            - seed (int): random seed for layout. Use same seed to reproduce layout, assuming same input data and cycles of simulation
            - seed_coords (pd.DataFrame): table of seed coordinates for each cluster in the base level. Must have columns 'id', 'x', 'y'
            - pull_base (bool): whether to allow circles in the base level to pull each other. Still respects radius_pad, just makes the base level more compact.
            Niche:
                - space_scale (float): multiplier for the space between circles on initial randomized layout.
                - expand_mult (float): multiplier for the expansion of circles. Somewhat unpredictable, can increase speed if too many circles start overlapping, but quite situational.
        Box2D simulation:
            - velocity_iterations (int): number of velocity iterations in the Box2D simulation
            - position_iterations (int): number of position iterations in the Box2D simulation
            - cycle_scale (int): multiplier for the number of cycles in the Box2D simulation per 50 child clusters
            - max_cycles (int): maximum number of cycles in the Box2D simulation
            - time_step (int): time step in the Box2D simulation

    Returns:
        dict: dictionary of circles at each level of clustering. Each level is a dictionary of circles, with the id of the circle as the key.
              Each circle object contains its xy position, radius, and size (number of sequences in the cluster)
    '''
    
    ### Process child levels in parallel
    
    size_dicts={}
    if len(levels)>1:
        process_args = []
        for lv, level_id in enumerate(levels[1:], start=1):
            df=clusters.loc[:,['id', level_id]]
            size=df.groupby(level_id).size().rename('size')#.reset_index()
            size_dicts[level_id]=dict(size)
            forced=size.index.isin(df[df['id'].isin(force_inclusion)][level_id].unique()) if force_inclusion is not None else None
            
            radii=GetRadii(size, minsize, 1, forced)
            print(f'Level {level_id}, clusters: {len(radii)}')
            seed_pos=None
            if seed_coords is not None:
                seed_pos=seed_coords.merge(df, on='id')
                seed_pos=seed_pos[seed_pos[level_id].isin(radii.index)]
                seed_pos=seed_pos.groupby(level_id)[['x','y']].mean()

            parent_level=levels[lv-1]
            # subset=clusters[clusters[level_id].isin(radii.index) & clusters[levels[lv-1]].isin(data[parent_level].keys())]
            subset=clusters[clusters[level_id].isin(radii.index)]
            
            print('preparing child clusters')
            for cluster, group in subset.groupby(parent_level):
                child_clusters = group[level_id].unique()
                # print(f'Level {lv} ({level_id}), parent: {cluster}, clusters: {len(group)}, children: {len(child_clusters)}')
                cycles=min( math.ceil(max(len(child_clusters)/50, 1) * cycle_scale), max_cycles )
                process_args.append((child_clusters, radii, radius_pad, space_scale, seed, seed_pos,
                                        cycles, None, pull, expand_mult,
                                        velocity_iterations, position_iterations, time_step, {'plevel':parent_level, 'parent':cluster,'level':level_id}))    

        print('starting pool')
        with Pool() as pool:
            results = list(pool.imap_unordered(process_child_clusters, process_args))
    else:
        results=[]

    ### Process base level to avoid collisions
    level_id=levels[0]
    df=clusters.loc[:,['id', level_id]]
    size=df.groupby(level_id).size().rename('size')#.reset_index()
    size_dicts[level_id]=dict(size)
    forced=size.index.isin(df[df['id'].isin(force_inclusion)][level_id].unique()) if force_inclusion is not None else None
    
    radii=GetRadii(size, minsize, 1, forced)
    radius_override={}
    if len(levels)>1:
        child_level=levels[1]
        children={}
        for circles, meta in results:
            if meta['level']==child_level:
                children[meta['parent']]=circles

        for cluster, radius in radii.items():
            if cluster not in children:
                continue
            child_circles=children[cluster]
            # get center of mass
            # x, y=CenterOfMass(child_circles)
            x0, x1, y0, y1=GetLimits(child_circles)
            cx=(x0+x1)/2
            cy=(y0+y1)/2
            max_dist=0
            # distance to each child circle + radius
            for c in child_circles:
                dist=((c.x-cx)**2+(c.y-cy)**2)**0.5 + c.r
                if dist>max_dist:
                    max_dist=dist
            if max_dist>radius:
                radii.loc[cluster]=max_dist
                radius_override[cluster]=radius

    print(f'Level {level_id}, clusters: {len(radii)}')
    seed_pos=None
    if seed_coords is not None:
        seed_pos=seed_coords.merge(df, on='id')
        seed_pos=seed_pos[seed_pos[level_id].isin(radii.index)]
        seed_pos=seed_pos.groupby(level_id)[['x','y']].mean()
    base_circles, _ =MakeLayout(radii, radius_pad=radius_pad, space_mult=space_scale, seed=seed, 
                               seed_coords=seed_pos, pull=pull if pull_base else 0, expand_mult=expand_mult,
                               velocity_iterations=velocity_iterations, position_iterations=position_iterations, time_step=time_step)
    
    for c in base_circles:
        if c.id in radius_override:
            c.r=radius_override[c.id]
        c.size=size_dicts[level_id][c.id]
    # collapse all data
    data={}
    data[level_id]=dict((c.id, c) for c in base_circles)
    grouped={}
    for circles, meta in results:
        for c in circles:
            c.size=size_dicts[meta['level']][c.id]
        if meta['level'] not in data:
            grouped[meta['level']]={meta['parent']:{'plevel':meta['plevel'], 'data':circles}}
            data[meta['level']]=dict((c.id, c) for c in circles)
        else:
            grouped[meta['level']][meta['parent']]={'plevel':meta['plevel'], 'data':circles}
            data[meta['level']].update((c.id, c) for c in circles)
    # for each level, recenter the circles into the parent circle
    for lv, level_id in enumerate(levels[1:], start=1):
        for parent, info in grouped[level_id].items():
            parent_circle=data[info['plevel']][parent]
            child_circles=info['data']
            x0, x1, y0, y1=GetLimits(child_circles)
            cx=(x0+x1)/2
            cy=(y0+y1)/2

            px, py=parent_circle.x, parent_circle.y
            for i, c in enumerate(child_circles):
                c.x=c.x-cx+px
                c.y=c.y-cy+py
    return data

def SaveLayout(data, outfile, config:SimpleNamespace=None):
    rows=[]
    for lv, c_dict in data.items():
        for c in c_dict.values():
            rows.append([lv, c.id, c.x, c.y, c.r])
    tb=pd.DataFrame(rows, columns=['level','cluster','x','y','r'])
    # print(tb)
    with open(outfile, 'w') as f:
        if config is not None:
            print(CommentedConfigString(config), file=f)
    tb.to_csv(outfile, mode='a', index=False)

def ReadLayout(file, config_check:SimpleNamespace=None):
    if not os.path.exists(file):
        return None

    if config_check is not None:
        saved_cstr=''
        with open(file) as f:
            for line in f:
                if not line.startswith('#'):
                    break
                saved_cstr+=line
        saved_config=ConfigFromComments(saved_cstr)
        # print(saved_config, config_check)
        if saved_config!=config_check:
            return None

    # re-use old layout
    tb=pd.read_csv(file, comment='#')
    data={}
    for lv, group in tb.groupby('level'):
        data[lv]={}
        for row in group.itertuples():
            data[lv][row.cluster]=Circle(row.x,row.y,row.r,row.cluster)
    return data
