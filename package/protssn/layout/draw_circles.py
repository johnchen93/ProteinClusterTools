import matplotlib.pyplot as plt

from layout.circle_collision import GetLimits

def DrawCircles(layout, out_label):
    '''Debug function to draw a layout of circles to a file'''
    all_c=[]
    for lv, c_dict in layout.items():
        all_c+=list(c_dict.values())
    xmin, xmax, ymin, ymax=GetLimits(all_c, 5)

    f, ax=plt.subplots(figsize=(10,10))
    ax.axis('off')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    for lv, c_dict in layout.items():
        lv_frac=lv/len(layout)
        for c in c_dict.values():
            # draw circle
            circle=plt.Circle((c.x, c.y), c.r, color='b', lw=.5, alpha=.2+lv_frac*.8, fill=False)
            ax.add_patch(circle)
    f.savefig(f'{out_label}_layout.png', dpi=300, bbox_inches='tight')