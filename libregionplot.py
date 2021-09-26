from functools import partial

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FuncFormatter
import matplotlib.patches as mpatches
import ipywidgets as widgets

def compute_criteria_arrays(results_filter, failed_to_stop='penalty'):
    """
    Return arrays of stop points suitable for the region plots
    """
    instances = {}
    accuracy = {}
    for dataset, results in results_filter.items():
        for cond, runs in results.items():
            instances.setdefault(cond, [])
            accuracy.setdefault(cond, [])
            for run in runs:
                if run[0] is not None:
                    try:
                        instances[cond].append(run[0])
                        accuracy[cond].append(run[1])
                    except KeyError:
                        # Criteria was excluded because it failed to stop
                        pass
                else:
                    if failed_to_stop == 'penalty':
                        # Penalize criteria which fail to stop so they are less likely to take places in the plot.
                        # TODO: Can we think of a more accurate penalty? Or do we want to just exclude?
                        max_inst = 0
                        min_acc = 1.
                        for rs in results.values():
                            for r in rs:
                                if r[0] is not None:
                                    max_inst = max(max_inst, r[0])
                                    min_acc = min(min_acc, r[1])
                        instances[cond].append(max_inst)
                        accuracy[cond].append(min_acc)
                    elif failed_to_stop == 'exclude':
                        if cond in instances:
                            del instances[cond]
                            del accuracy[cond]
                    elif failed_to_stop == 'include':
                        pass
                    else:
                        raise Exception(f'invalid failed_to_stop value {failed_to_stop}')

    instances_mean = []
    accuracy_mean = []
    instances_upper = []
    instances_lower = []
    accuracy_upper = []
    accuracy_lower = []
    for cond in instances.keys():
        if len(instances[cond]) > 0:
            instances_mean.append(np.mean(instances[cond]))
            accuracy_mean.append(np.mean(accuracy[cond]))

            instances_upper.append(np.percentile(instances[cond], 97.5))
            accuracy_upper.append(np.percentile(accuracy[cond], 2.5))

            instances_lower.append(np.percentile(instances[cond], 97.5))
            accuracy_lower.append(np.percentile(accuracy[cond], 2.5))
            
    return np.array(list(instances.keys())), np.array(instances_mean), np.array(accuracy_mean), np.array(instances_upper), np.array(accuracy_upper), np.array(instances_lower), np.array(accuracy_lower)


def C_rel_nonvec(accuracy, instances, A, l):
    return (1-accuracy)*A+instances*l


C_rel = np.vectorize(C_rel_nonvec, signature="(),(),(a,b),(a,b)->(a,b)")


def make_grid():
    """
    Return a grid on which to plot the criterias' performance
    """
    A, l = np.mgrid[0:1e6:1000j,0:1e2:1000j]
    A = A.T
    l = l.T
    return A, l


def eval_on_grid(A, l, conds, instances_mean, accuracy_mean, instances_upper, accuracy_upper, instances_lower, accuracy_lower):
    mean_grid = C_rel(accuracy_mean, instances_mean, A, l)
    # highest cost is lowest accuracy & most instances
    upper_grid = C_rel(accuracy_lower, instances_upper, A, l)
    # lowest cost is highest accuracy & fewest instances
    lower_grid = C_rel(accuracy_upper, instances_lower, A, l)

    minimized_mean = np.argmin(mean_grid, axis=0)
    minimized_upper = np.argmin(upper_grid, axis=0)
    minimized_lower = np.argmin(lower_grid, axis=0)
    
    conds_indeterminate = np.append(conds, 'Indeterminate')
    
    # Set regions where criteria are not statistically different from one another to an indeterminate color
    # TODO: Represent this with alpha instead?
    minimized_error = np.copy(minimized_mean)
    minimized_error[minimized_upper!=minimized_lower] = conds_indeterminate.shape[0]-1
    
    return conds_indeterminate, minimized_error, mean_grid
    

def plot_regions(results_filter, A, l, conds_indeterminate, minimized_error, ax=None, colors=None, title=None, figsize=(10,6), patches=None, left=True, bottom=True):
    
    min_ids = np.unique(minimized_error)
    min_keys = conds_indeterminate[min_ids]

    # Map results to consecutive integers, this stops matplotlib plotting the colorbar as if it was continuous
    minimized_mapped = np.copy(minimized_error) 
    for new, old in enumerate(min_ids):
        minimized_mapped[minimized_error==old] = new
    mapped_ids = np.arange(len(min_ids))
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        
    if colors is not None:
        cmap = ListedColormap([colors[key] for key in min_keys])
    else:
        cmap = ListedColormap(sns.color_palette("pastel", len(min_ids)).as_hex())
        
    im = ax.imshow(minimized_mapped, origin='lower', cmap=cmap)
    
    # Axis labels, complicated because matplotlib isn't intended for mapped
    # image pixel labels
    xtickspace = np.where(np.isclose(A[0,:], np.linspace(0, 1e6, 5).reshape(-1, 1), atol=500))[1]
    xtickspace = xtickspace[xtickspace!=499] # hack to fix broken tolerance
    ytickspace = np.where(np.isclose(l[:,0], np.linspace(0, 100, 5).reshape(-1, 1), atol=1e-1))[1]
    ax.set_xticks(xtickspace)
    def formatter(x, pos):
        val = A[0,xtickspace][pos]/A[0,xtickspace][-1]*100
        if True:
            return f"{val:.0f}"
        else:
            return f"{val:.1f}"
    mformatter = FuncFormatter(formatter)
    mformatter.set_offset_string(fr"$\times 10^{{{np.log10(A[0,xtickspace][-1])-2:.0f}}}$")
    ax.xaxis.set_major_formatter(mformatter)
    ax.set_yticks(ytickspace)
    ax.set_yticklabels([f"{x:.0f}" for x in l[ytickspace,0]])
    
    #ax.grid(alpha=0.7)
    
    if left:
        ax.set_ylabel('$l$')
    if bottom:
        ax.set_xlabel('$nm$')
        
    ax.set_title(title if title is not None else 'Cost-Optimal Stopping Criteria')
    
    if patches is None:
        cb = plt.colorbar(im, spacing='uniform')
        cb.set_ticks(mapped_ids)
        cb.set_ticklabels(min_keys)
    else:
        values = np.unique(minimized_mapped.ravel())
        assert np.array_equal(values, mapped_ids)
        c = [im.cmap(im.norm(value)) for value in values]
        existing = {patch.get_label() for patch in patches}
        patches.extend([
            mpatches.Patch(color=c[i], label=min_keys[i]) for i in range(values.shape[0]) if min_keys[i] not in existing
        ])
    
    
def regions(results_filter, failed_to_stop='penalty', ax=None, colors=None, title=None, figsize=(10,6), patches=None, left=True, bottom=True):
    conds, instances_mean, accuracy_mean, instances_upper, accuracy_upper, instances_lower, accuracy_lower = compute_criteria_arrays(
        results_filter, failed_to_stop=failed_to_stop
    )
    A, l = make_grid()
    conds_indeterminate, minimized_error, mean_grid = eval_on_grid(
        A, l, conds, instances_mean, accuracy_mean, instances_upper, accuracy_upper, instances_lower, accuracy_lower
    )
    plot_regions(results_filter, A, l, conds_indeterminate, minimized_error, ax=ax, colors=colors, title=title, figsize=figsize, patches=patches, left=left, bottom=bottom)
    
    
def costs(results_filter, failed_to_stop='penalty'):
    conds, instances_mean, accuracy_mean, instances_upper, accuracy_upper, instances_lower, accuracy_lower = compute_criteria_arrays(
        results_filter, failed_to_stop=failed_to_stop
    )
    A, l = make_grid()
    conds_indeterminate, minimized_error, mean_grid = eval_on_grid(
        A, l, conds, instances_mean, accuracy_mean, instances_upper, accuracy_upper, instances_lower, accuracy_lower
    )
    plot_costs(A, l, conds, instances_mean, mean_grid)
    
    
def plot_costs(A, l, conds, instances_mean, mean_grid):
    # TODO: Make the required data
    N_PER_ROW = 6
    fig, axes = plt.subplots(int(np.ceil(len(instances_mean)/N_PER_ROW)), N_PER_ROW, figsize=(20,14))
    
    _min = np.min(mean_grid)
    _max = np.max(mean_grid)
    
    for i, (key, ax) in enumerate(zip(conds, axes.flatten())):
        contours = ax.imshow(mean_grid[i], cmap='coolwarm', origin='lower', vmin=_min, vmax=_max)
        ax.set_ylabel('$l$')
        ax.set_xlabel('$nm$')
        xtickspace = np.linspace(0, A.shape[1]-1, 4, dtype=int)
        ytickspace = np.linspace(0, A.shape[0]-1, 6, dtype=int)
        ax.set_xticks(xtickspace)
        ax.set_xticklabels([f"{round(x, -4):.0f}" for x in A[0,xtickspace]])
        ax.set_yticks(ytickspace)
        ax.set_yticklabels([f"{x:.0f}" for x in l[ytickspace,0]])
        ax.set_title(key)
    for i in range(len(instances_mean), len(axes.flatten())):
        axes.flatten()[i].remove()
    #plt.colorbar(contours)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.25, 0.12, 0.65, 0.05])
    fig.colorbar(contours, cax=cbar_ax, orientation='horizontal')
    plt.tight_layout()

 
def optimal_for_params(conds, accuracy_mean, instances_mean, results_filter, n, m, l, title, average=False):
    values = C_rel_nonvec(accuracy_mean, instances_mean, n*m, l)
    minimum = np.argmin(values)
    print(f"The optimal criteria is {conds[minimum]} with cost ${values[minimum]:.2f}")
    
    print(f"The ranking of all criteria is {conds[np.argsort(values)]}")
    
    def autorank_func(x, accuracy, f1, roc_auc):
        #print("in", x, accuracy, f1, roc_auc)
        ret = C_rel_nonvec(accuracy, x, n*m, l)
        #print(ret)
        return ret
    
    from libstop import rank_stop_conds
    rank_stop_conds(
        results_filter, 
        "func",
        func=autorank_func,
        title=title,
        average=average
    )


def interactive_explore_cost(results_filter, failed_to_stop='penalty', title='Example', average=False):
    n_widget = widgets.IntText(
        value=87600,
        description='Expected number of misclassifications n:',
        disabled=False
    )
    m_widget = widgets.FloatText(
        value=0.0001,
        description='Cost of misclassifications m ($):',
        disabled=False
    )
    l_widget = widgets.FloatText(
        value=0.129,
        description='Cost of labels l ($):',
        disabled=False
    )
    
    conds, instances_mean, accuracy_mean, instances_upper, accuracy_upper, instances_lower, accuracy_lower = compute_criteria_arrays(
        results_filter, failed_to_stop=failed_to_stop
    )
    
    widgets.interact(
        optimal_for_params, 
        conds=widgets.fixed(conds), 
        accuracy_mean=widgets.fixed(accuracy_mean), 
        instances_mean=widgets.fixed(instances_mean),
        results_filter=widgets.fixed(results_filter),
        title=widgets.fixed(title),
        average=widgets.fixed(False),
        n=n_widget, 
        m=m_widget, 
        l=l_widget)
    