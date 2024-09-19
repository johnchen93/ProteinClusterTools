import pandas as pd

def AnnotateClusters(clusters, levels, annot_table, column, cluster_key, annot_key,  numeric_func='mean', dropna=False):
    '''
    Function to aggregate annotations at the cluster level.

    Arguments:
    - clusters: DataFrame with cluster assignments, with each level of clustering as its own column
    - levels: List of levels (columns) to aggregate annotations within 'clusters'
    - annot_table: DataFrame with annotations to aggregate
    - column: Column in 'annot_table' to aggregate
    - cluster_key: Column in 'clusters' that contains the sequence ID
    - annot_key: Column in 'annot_table' that contains the sequence ID
    - numeric_func: Function to aggregate numeric data, default is 'mean'. Can be any string or function accepted by the Pandas.groupby.agg function.
    - dropna: Whether to drop NA values when aggregating numeric data. Default is False.

    Returns a dictionary with the following keys
    - method: The method used to aggregate the data
    - label: The label of the annotation
    - levels: List of levels
    - value: The column in 'annot_table' that was aggregated, useful reference for labelling legends
    - For each level in 'levels', a dictionary with the following keys:
        - id: The cluster ID
        - value: The aggregated value    
    '''

    is_numeric=pd.api.types.is_numeric_dtype(annot_table[column])
    method=numeric_func if is_numeric else 'counts'
    out={'method':method, 'label':f'{column}_{method}', 'levels':levels, 'value':column}
    for level in levels:
        sel=clusters[[cluster_key,level]].copy()
        data=sel.merge(annot_table[[annot_key, column]], left_on=cluster_key, right_on=annot_key)
        annot=data.groupby(level)
        
        if method!='counts':
            annot=annot[column].agg(numeric_func)
        else:
            annot=annot[column].value_counts(dropna=dropna)

        out[level]=annot.reset_index().rename(columns={level:'id', column:'value'})
    
    return out

import colorsys

def ColorSaturation(rgba, saturation):
    """
    Desaturate an RGB color.

    :param rgb: A tuple of (R, G, B) in the range 0-255.
    :param desaturation_factor: Float between 0 (no change) and 1 (fully desaturated).
    :return: Desaturated RGB color, as a tuple in the range 0-255.
    """
    # Normalize RGB values to 0-1 range
    r, g, b = [x  for x in rgba[:3]]

    # Convert RGB to HSV
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    # Reduce the saturation
    s = saturation

    # Convert back to RGB
    desaturated_rgb = colorsys.hsv_to_rgb(h, s, v)

    # Convert back to 0-255 range
    return [x for x in desaturated_rgb]+[rgba[3]]

import matplotlib as mpl

def AdjustColormapSaturation(cmap, saturation=1):
    """
    Adjust the saturation of a Matplotlib colormap.

    :param cmap: Original Matplotlib colormap.
    :param saturation_factor: Saturation factor (0-1, where 1 is full saturation).
    :return: New colormap with adjusted saturation.
    """
    # Create a new colormap array based on the original colormap
    new_cmap_data = cmap(range(cmap.N))

    # Adjust the saturation for each color in the colormap
    for i, rgba in enumerate(new_cmap_data):
        # Convert RGBA to HSV
        h, s, v = colorsys.rgb_to_hsv(rgba[0], rgba[1], rgba[2])
        # Adjust saturation
        s *= saturation
        # Convert back to RGB
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        # Update the new colormap array
        new_cmap_data[i] = r, g, b, rgba[3]

    # Create a new colormap from the adjusted array
    return mpl.colors.ListedColormap(new_cmap_data)

def AssignCategoricalColors(all_cats, color_map, shuffle_colors_seed=None, is_categorical=True, top_n=None):

    include=all_cats if top_n is None else all_cats[:top_n]

    # make a color map, give each category a color
    c_mapping={}
    for i, cat in enumerate(include):
        if is_categorical:
            c_mapping[cat]=color_map(i%color_map.N)
        else:
            c_mapping[cat]=color_map(i/(len(include)-1))
    
    if shuffle_colors_seed is not None:
        # seed random number generator
        Random=random.Random(shuffle_colors_seed)
        # shuffle the colors
        k, v=[list(x) for x in zip(*c_mapping.items())]
        Random.shuffle(v)
        c_mapping=dict(zip(k,v))
    
    return c_mapping
        
import random
c_template='rgba({},{},{},{})'
def FormatColor(color, format='bokeh', scale=255):
    '''
    Args:
        color: Iterable of RGBA values in the range 0-1.
        format: 'bokeh', 'hex' or 'tuple'
        scale: Scale to multiply the color values by. Default is 255, assuming input is 0-1.
    '''
    color=[int(x*scale) for x in color]
    if format=='bokeh':
        return c_template.format(*color)
    elif format=='hex':
        return "#{:02x}{:02x}{:02x}".format(*color[:3])
    elif format=='tuple':
        return tuple(color)


def ColorAnnot( annot:dict, cmap='viridis', top_n=None, saturation=1, shuffle_colors_seed=None, 
                vmin=None, vmax=None, cmap_is_categorical=True, binary_blend=False, color_format='bokeh', direct_map_colors=False, annot_order=None):
    '''
    Colors a set of annotations made with AnnotateClusters.

    Arguments:
    - annot: Dictionary with annotations, as returned by AnnotateClusters
    - cmap: Colormap to use. Can be a string (name of a Matplotlib colormap) or a Matplotlib colormap object.
    - top_n: Number of top categories to color. Default is None, which colors all categories.
    - saturation: Saturation of the colormap. Default is 1.
    - shuffle_colors_seed: Seed for shuffling colors in categorical data. Mostly useful when there are many categories and there is no preference for color order.
                         Default is None, which does not shuffle colors. 
    - vmin: Minimum value for a numeric colormap. Default is None, which uses the minimum value in the data.
    - vmax: Maximum value for a numeric colormap. Default is None, which uses the maximum value in the data.
    - cmap_is_categorical: When a colormap object is given (instead of a string), if the colormap should be considered categorical. Only applies for categorical data. Default is True.
    - binary_blend: For blending colors. If True, treat all instances in categorical data to be yes/no (i.e., max count of 1 if present). Default is False.
    - color_format: Format of the color to output. Can be 'bokeh', 'hex' or 'tuple'. Default is 'bokeh'.
    - direct_map_colors: If True, map the color directly to the category (assumes only one color per id). Default is False, which blends the colors based on the count of each category.
    - annot_order: Order of annotations for categorical data. Default is None, which orders by overall count in the dataset.

    Returns a dictionary with the following:
    - value: The column in 'annot_table' that was aggregated, useful reference for labelling legends
    - method: The method used to aggregate the data
    - min: Minimum value in the data
    - max: Maximum value in the data
    - colorscale: List of colors in the colormap
    - categories: Dictionary with categories as keys and colors as values. Only present if the data is categorical.
    - For each level in 'levels', a dictionary with the following keys:
        - id: The cluster ID
        - value: The color of the cluster
        
    '''

    # settle color scheme
    if type(cmap)==str:
        color_map=mpl.colormaps.get_cmap(cmap)
        is_categorical=cmap in ['tab10', 'tab20', 'tab20b', 'tab20c','Pastel1', 'Pastel2', 'Paired', 'Set1', 'Set2', 'Set3','Accent','Dark2']
    else:
        color_map=cmap
        is_categorical=cmap_is_categorical
    color_map=AdjustColormapSaturation(color_map, saturation)

    colors={'value':annot['value'], 'method':annot['method']}
    if annot['method']!='counts':
        # determine range of values
        # Normalize data (scale it to range between 0 and 1)
        min_value=min([min(annot[x]['value']) for x in annot['levels']]) if vmin is None else vmin
        max_value=max([max(annot[x]['value']) for x in annot['levels']]) if vmax is None else vmax

        # Choose a color map
        cmap_resolution=100
        colors['colorscale'] = [[i/(cmap_resolution-1), FormatColor(color_map(i/(cmap_resolution-1)), color_format)] for i in range(cmap_resolution)]
        colors['min']=min_value
        colors['max']=max_value
        
        for level in annot['levels']:
            data=annot[level].copy()
            values=data['value']
            normalized_data = (values - min_value) / (max_value - min_value)

            # Map the normalized data to colors from the color map
            mapped_colors = color_map(normalized_data)

            # format rgba string
            mapped_colors=[FormatColor(c, color_format) for c in mapped_colors]

            # save colors alongside with id
            colors[level]=dict(zip(data['id'], mapped_colors))
    else:
        # get all categories
        
        all_levels=[]
        for level in annot['levels']:
            all_levels.append(annot[level])
        
        if annot_order is not None:
            all_cats=annot_order
        else:
            all_level_df=pd.concat(all_levels).dropna(subset=['value']) # ignore NA values when distributing colors
            # all categories in order of count
            all_cats=all_level_df.groupby('value')['count'].sum().reset_index().sort_values('count', ascending=False)['value'].values
        c_mapping=AssignCategoricalColors(all_cats, color_map, shuffle_colors_seed, is_categorical, top_n)
        colors['categories']={k:FormatColor(v, color_format) for k,v in c_mapping.items()}
        
        for level in annot['levels']:
            data=annot[level].copy()
            # map the category to the color
            data['color']=data['value'].map(c_mapping)
            # blend the colors by weight
            blended={}
            if direct_map_colors:
                data.dropna(subset=['color'], inplace=True)
                colors[level]=dict(zip(data['id'], data['color'].apply(lambda x: FormatColor(x, color_format))))
                continue
            for group, subdata in data.groupby('id'):
                r,g,b=0,0,0
                total=subdata['count'].sum() if not binary_blend else len(subdata)
                for row in subdata.itertuples():
                    weight=row.count if not binary_blend else 1
                    if row.value not in c_mapping: 
                        r+=weight
                        g+=weight
                        b+=weight
                    else:
                        r+=row.color[0]*weight
                        g+=row.color[1]*weight
                        b+=row.color[2]*weight
                    
                if total==0:
                    blended[group]='rgba(0,0,0,0)'
                else:
                    r/=total
                    g/=total
                    b/=total
                    blended[group]=FormatColor((r,g,b,1), color_format)
            colors[level]=blended
        
    return colors

### Unused below
def OutlineClusters(clusters, levels, targets, rep_map=None, mapping:dict=None, map_func=None, color='red', label='selected', alpha=.7):
    '''
    rep_map: Dataframe with 'rep' and 'id' columns. Assuming each sequence in the clusters table is a representative of a larger set, this table maps the representative to the larger set. 
                Search of targets is on larger set, instead.
    '''
    clusters=clusters.copy()
    if mapping is not None:
        clusters['id']=clusters['id'].map(mapping)
    if map_func is not None:
        clusters['id']=clusters['id'].apply(lambda x: map_func(x))
    if rep_map is not None:
        clusters=clusters.rename(columns={'id':'rep'}).merge(rep_map, on='rep')
    color=mpl.colors.to_rgba(color)
    color=color[:3]+(alpha,)
    out={'color':c_template.format(*[int(x*255) for x in color]), 'levels':levels, 'label':label}
    for level in levels:
        # out[level]=clusters[clusters['id'].isin(targets)][level].unique().tolist()
        # save table indicating which targets were in which cluster
        # out[level]=clusters[clusters['id'].isin(targets)][['id', level]].groupby(level)['id'].apply(list).reset_index().rename(columns={level:'id','id':'value'})
        out[level]=clusters[clusters['id'].isin(targets)][['id', level]].groupby(level)['id'].apply(list).to_dict()
    return out

def OutlineClustersDirect(clusters, level, color='red'):
    return {level:clusters, 'color':mpl.colors.to_rgba(color), 'levels':[level]}