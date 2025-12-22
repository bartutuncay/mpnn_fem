import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker
import networkx as nx
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import warnings
import seaborn as sns

# set scientific formatter
def set_to_sci_format(ax_axis):
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax_axis.set_major_formatter(formatter)

def get_vmin_vmax(graph_object, node_type, channel):
    vmin = np.nanmin([graph_object.x[graph_object.node_type == node_type][:, channel].min(), graph_object.y[graph_object.node_type == node_type][:, channel].min()])
    vmax = np.nanmax([graph_object.x[graph_object.node_type == node_type][:, channel].max(), graph_object.y[graph_object.node_type == node_type][:, channel].max()])
    return vmin, vmax

def plot_mesh(graph_object:Data, plot_type:str, plot_predicted=False, plot_diff=False, plot_separate=False, xlimits=None, ylimits=None,
              add_farfield_info=True, fig_size=None, tile_vertical=False, show=True):
    assert plot_type in {'graph', 'pressure', 'velocity_mag', 'velocity_x', 'velocity_y', 'sdf', 'surface_pressure'}

    if fig_size is None:
        fig_size = (12,8)

    # send object to cpu and gather bounds of the domain
    graph_object = graph_object.cpu()
    unique_points = torch.unique(graph_object.triangles)
    if xlimits is not None:
        xmin = xlimits[0]
        xmax = xlimits[1]
    else:
        xmin = graph_object.triangle_points[0, unique_points].min()
        xmax = graph_object.triangle_points[0, unique_points].max()

    if ylimits is not None:
        ymin = ylimits[0]
        ymax = ylimits[1]
    else:
        ymin = graph_object.triangle_points[1, unique_points].min()
        ymax = graph_object.triangle_points[1, unique_points].max()

    if  plot_predicted and plot_type != 'graph':
        if plot_separate:
            tile_vertical = False
            # plot two figures with real and predicted
            fig1, ax = plt.subplots(figsize=fig_size, nrows=1, ncols=1)
            fig2, ax_p = plt.subplots(figsize=fig_size, nrows=1, ncols=1)
            fig = [fig1, fig2]
            if plot_diff:
                fig3, ax_d = plt.subplots(figsize=fig_size, nrows=1, ncols=1)
                fig.append(fig3)
        else:
            if tile_vertical:
                # plot vertically  figures of real vs predicted and difference (if enabled)
                nrows = 3 if plot_diff else 2
                fig, axs = plt.subplots(figsize=fig_size, nrows=nrows, ncols=1)
                ax = axs[0]
                ax_p = axs[1]
                if plot_diff:
                    ax_d = axs[2]
                
            else:
                # plot side by side figures of real vs predicted and difference (if enabled)
                ncols = 3 if plot_diff else 2
                fig, axs = plt.subplots(figsize=fig_size, nrows=1, ncols=ncols)
                ax = axs[0]
                ax_p = axs[1]
                if plot_diff:
                    ax_d = axs[2]
    else:
        fig, ax = plt.subplots(figsize=fig_size)

    triangulation = tri.Triangulation(x=graph_object.triangle_points[0, :], y=graph_object.triangle_points[1, :],
                                      triangles=graph_object.triangles)

    # plot graph and mesh
    if plot_type == 'graph':
        if graph_object.num_nodes > 10000:
            warnings.warn('Plotter: this is a big graph, why not get some coffee while you wait')
        G = to_networkx(graph_object)
        node_pos_dict = {}
        for i in range(graph_object.num_nodes):
            node_pos_dict[i] = graph_object.pos[i,:].tolist()
        nx.draw_networkx(G, pos=node_pos_dict, node_color = graph_object.node_type,node_size=20,linewidths=6)

    # plot mesh with pressure
    elif plot_type == 'pressure':
        if plot_predicted:
            vmin, vmax = get_vmin_vmax(graph_object, 0, 0)
            tpc_p = ax_p.tripcolor(triangulation, graph_object.x[graph_object.node_type == 0, 0], shading='flat', vmin=vmin, vmax=vmax)
            if tile_vertical:
                cb_p = plt.colorbar(tpc_p, ax = axs.ravel().tolist())
            else:
                cb_p = plt.colorbar(tpc_p, ax=ax_p)
            cb_p.ax.set_ylabel('Pressure [Pa]', fontsize=16)
            cb_p.ax.yaxis.set_label_coords(4.5, 0.5)
            set_to_sci_format(cb_p.ax.yaxis)
            if plot_diff:
                tpc_d = ax_d.tripcolor(triangulation, graph_object.y[graph_object.node_type == 0, 0] - graph_object.x[graph_object.node_type == 0, 0], shading='flat')
                if tile_vertical:
                    cb_d = plt.colorbar(tpc_d, ax=axs.ravel().tolist())
                else:
                    cb_d = plt.colorbar(tpc_d, ax=ax_d)
                cb_d.ax.set_ylabel('Pressure Difference [Pa]', fontsize=16)
                cb_d.ax.yaxis.set_label_coords(4.5, 0.5)
                set_to_sci_format(cb_d.ax.yaxis)
            
        else:
            vmin, vmax = None, None
        tpc = ax.tripcolor(triangulation, graph_object.y[graph_object.node_type == 0, 0], shading='flat', vmin=vmin, vmax=vmax)
        if not tile_vertical:
            cb = plt.colorbar(tpc, ax=ax)
            cb.ax.set_ylabel('Pressure [Pa]', fontsize=16)
            cb.ax.yaxis.set_label_coords(4.5, 0.5)
            set_to_sci_format(cb.ax.yaxis)

    # plot mesh with velocity magnitude
    elif plot_type == 'velocity_mag':
        if plot_predicted:
            vmin, vmax = get_vmin_vmax(graph_object, 0, (1,2))
            tpc_p = ax_p.tripcolor(triangulation, torch.norm(graph_object.x[graph_object.node_type == 0, 1:3], dim=1), shading='flat', vmin=vmin, vmax=vmax)
            if tile_vertical:
                cb_p = plt.colorbar(tpc_p, ax=axs.ravel().tolist())
            else:
                cb_p = plt.colorbar(tpc_p, ax=ax_p)
            cb_p.ax.set_ylabel('Velocity Magnitude [m/s]', fontsize=16)
            cb_p.ax.yaxis.set_label_coords(4.5, 0.5)
            if plot_diff:
                tpc_d = ax_d.tripcolor(triangulation, torch.norm(graph_object.y[graph_object.node_type == 0, 1:3], dim=1) - torch.norm(graph_object.x[graph_object.node_type == 0, 1:3], dim=1), shading='flat')
                if tile_vertical:
                    cb_d = plt.colorbar(tpc_d, ax=axs.ravel().tolist())
                else:
                    cb_d = plt.colorbar(tpc_d, ax=ax_d)
                cb_d.ax.set_ylabel('Velocity Magnitude Difference [m/s]', fontsize=16)
                cb_d.ax.yaxis.set_label_coords(4.5, 0.5)
                set_to_sci_format(cb_d.ax.yaxis)
            
        else:
            vmin, vmax = None, None
        tpc = ax.tripcolor(triangulation, torch.norm(graph_object.y[graph_object.node_type == 0, 1:3], dim=1), shading='flat', vmin=vmin, vmax=vmax)
        if not tile_vertical:
            cb = plt.colorbar(tpc, ax=ax)
            cb.ax.set_ylabel('Velocity Magnitude [m/s]', fontsize=16)
            cb.ax.yaxis.set_label_coords(4.5, 0.5)

    # plot mesh with x velocity
    elif plot_type == 'velocity_x':
        if plot_predicted:
            vmin, vmax = get_vmin_vmax(graph_object, 0, 1)
            tpc_p = ax_p.tripcolor(triangulation, graph_object.x[graph_object.node_type == 0, 1], shading='flat', vmin=vmin, vmax=vmax)
            if tile_vertical:
                cb_p = plt.colorbar(tpc_p, ax=axs.ravel().tolist())
            else:
                cb_p = plt.colorbar(tpc_p, ax=ax_p)
            cb_p.ax.set_ylabel('X-Velocity [m/s]', fontsize=16)
            cb_p.ax.yaxis.set_label_coords(4.5, 0.5)
            if plot_diff:
                tpc_d = ax_d.tripcolor(triangulation, graph_object.y[graph_object.node_type == 0, 1] - graph_object.x[graph_object.node_type == 0, 1], shading='flat')
                if tile_vertical:
                    cb_d = plt.colorbar(tpc_d, ax=axs.ravel().tolist())
                else:
                    cb_d = plt.colorbar(tpc_d, ax=ax_d)
                cb_d.ax.set_ylabel('X-Velocity Difference [m/s]', fontsize=16)
                cb_d.ax.yaxis.set_label_coords(4.5, 0.5)
                set_to_sci_format(cb_d.ax.yaxis)
        else:
            vmin, vmax = None, None
        tpc = ax.tripcolor(triangulation, graph_object.y[graph_object.node_type == 0, 1], shading='flat', vmin=vmin, vmax=vmax)
        if not tile_vertical:
            cb = plt.colorbar(tpc, ax=ax)
            cb.ax.set_ylabel('X-Velocity [m/s]', fontsize=16)
            cb.ax.yaxis.set_label_coords(4.5, 0.5)


    # plot mesh with x velocity
    elif plot_type == 'velocity_y':
        if plot_predicted:
            vmin, vmax = get_vmin_vmax(graph_object, 0, 2)
            tpc_p = ax_p.tripcolor(triangulation, graph_object.x[graph_object.node_type == 0, 2], shading='flat', vmin=vmin, vmax=vmax)
            if tile_vertical:
                cb_p = plt.colorbar(tpc_p, ax=axs.ravel().tolist())
            else:
                cb_p = plt.colorbar(tpc_p, ax=ax_p)
            cb_p.ax.set_ylabel('Y-Velocity [m/s]', fontsize=16)
            cb_p.ax.yaxis.set_label_coords(4.5, 0.5)
            if plot_diff:
                tpc_d = ax_d.tripcolor(triangulation, graph_object.y[graph_object.node_type == 0, 2] - graph_object.x[graph_object.node_type == 0, 2], shading='flat')
                if tile_vertical:
                    cb_d = plt.colorbar(tpc_d, ax=axs.ravel().tolist())
                else:
                    cb_d = plt.colorbar(tpc_d, ax=ax_d)
                cb_d.ax.set_ylabel('Y-Velocity Difference [m/s]', fontsize=16)
                cb_d.ax.yaxis.set_label_coords(4.5, 0.5)
                set_to_sci_format(cb_d.ax.yaxis)
        else:
            vmin, vmax = None, None
        tpc = ax.tripcolor(triangulation, graph_object.y[graph_object.node_type == 0, 2], shading='flat', vmin=vmin, vmax=vmax)
        if not tile_vertical:
            cb = plt.colorbar(tpc, ax=ax)
            cb.ax.set_ylabel('Y-Velocity [m/s]', fontsize=16)
            cb.ax.yaxis.set_label_coords(4.5, 0.5)
            
    # plot mesh with signed distance function
    elif plot_type == 'sdf':
        tpc = ax.tripcolor(triangulation, graph_object.x[graph_object.node_type == 0, 1], shading='flat', cmap=sns.color_palette("flare", as_cmap=True))
        if not tile_vertical:
            cb = plt.colorbar(tpc, ax=ax)
            cb.ax.set_ylabel('Signed Distance [m]', fontsize=16)
            cb.ax.yaxis.set_label_coords(4.5, 0.5)


    elif plot_type == 'surface_pressure':
        airfoil_nodes = graph_object.node_type == 1
        if plot_predicted:
            vmin, vmax = get_vmin_vmax(graph_object, 1, 0)
            scatter_p = ax_p.scatter(graph_object.pos[airfoil_nodes, 0], graph_object.pos[airfoil_nodes, 1], c=graph_object.x[airfoil_nodes, 0], vmin=vmin, vmax=vmax)
            cb_p = plt.colorbar(scatter_p, ax=ax_p)
            cb_p.ax.set_ylabel('Pressure [Pa]', fontsize=16)
            cb_p.ax.yaxis.set_label_coords(3, 0.5)
            set_to_sci_format(cb_p.ax.yaxis)
            if plot_diff:
                scatter_d = ax_d.scatter(graph_object.pos[airfoil_nodes, 0], graph_object.pos[airfoil_nodes, 1], c=graph_object.y[airfoil_nodes, 0] - graph_object.x[airfoil_nodes, 0], vmin=vmin, vmax=vmax)
                cb_d = plt.colorbar(scatter_d, ax=ax_d)
                cb_d.ax.set_ylabel('Pressure Difference [Pa]', fontsize=16)
                cb_d.ax.yaxis.set_label_coords(3, 0.5)
                set_to_sci_format(cb_d.ax.yaxis)
        else:
            vmin, vmax = None, None
        scatter = ax.scatter(graph_object.pos[airfoil_nodes, 0], graph_object.pos[airfoil_nodes, 1] , c=graph_object.y[airfoil_nodes, 0], vmin=vmin, vmax=vmax)
        cb = plt.colorbar(scatter, ax=ax)
        cb.ax.set_ylabel('Pressure [Pa]', fontsize=16)
        cb.ax.yaxis.set_label_coords(3, 0.5)
        set_to_sci_format(cb.ax.yaxis)

    ax.triplot(triangulation, color='black', alpha=0.2)
    ax.axis('equal')
    ax.set_aspect('equal')
    ax.set_title('Ground Truth')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    if plot_predicted and plot_type != 'graph':
        ax_p.triplot(triangulation, color='black', alpha=0.2)
        ax_p.axis('equal')
        ax_p.set_aspect('equal')
        ax_p.set_title('Predicted')
        ax_p.set_xlim(xmin, xmax)
        ax_p.set_ylim(ymin, ymax)
        if plot_diff:
            ax_d.triplot(triangulation, color='black', alpha=0.2)
            ax_d.axis('equal')
            ax_d.set_aspect('equal')
            ax_d.set_title('Difference')
            ax_d.set_xlim(xmin, xmax)
            ax_d.set_ylim(ymin, ymax)
            
    if add_farfield_info:
        Umag = graph_object.globals_y[0,0].item()
        aoa = graph_object.globals_y[0,1].item()
        Ti = graph_object.globals_y[0,2].item()
        Re = Umag / 1.511e-05
        Ma = Umag / 340.294
        fig.suptitle(f'Umag = {Umag:.2f} m/s, AoA = {aoa:.2f} deg, Ti = {Ti:.2f} %, Re = {Re:.2e}, Ma = {Ma:.2f}', fontsize=16)
    
    if show:
        plt.show()

    return fig