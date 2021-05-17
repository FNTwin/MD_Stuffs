import argparse
import os
import sys
import glob
from numpy import linspace, array, loadtxt, meshgrid, isnan, where, concatenate, pi
from numpy import amin, amax
from scipy.interpolate import griddata, Rbf
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

# Esempio di comando per eseguirlo:
# python map_vor_v0.1.py -i /Users/gab/Desktop/watermaps/solvent_accessibility.stride5_dist18.00_samples200.txt --vo true
# Il Path di output ora e` di default quello dove si sta eseguendo il file! Inoltre e` possibile passare piu input allo stesso tempo anche
# da cartelle differenti (senza path cerca dove si sta eseguendo il file): es: -i file1.txt C:\Users\file2.txt file3.txt
# Per la produzione massica di grafici basta passare il path di una o piu cartelle, es: -i directory1\ directory2
# Lo script cercherà automaticamente i file .txt presenti in queste directory
# Ci sono varie flag, per vederle utilizzare la flag --help
# Per eseguire lo script da spyder fare (CTRL + F6): >run>configuration per file
# In General Setting abilitare Command Line Option inserendo i vari comandi nel riquadro a destra dell'opzione
# Infine eseguire il file tramite run


def main(in_file, args):
    n_countourf = 400

    basename = os.path.basename(in_file).rsplit(
        '.', 1)[0]  # Remove the file format
    t, c, v = (array(args.col)-1)
    arr = loadtxt(in_file)
    theta, cosphi, valore = arr[:, t], arr[:, c], arr[:, v]

    # Interpolation of the data points to construct the 2D heatmap
    x, y, z = theta, cosphi, valore
    xi = linspace(-pi, pi, args.n)
    yi = linspace(-1, 1, args.n)
    X, Y = meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic', rescale="true")
    rbf3 = Rbf(x, y, z, function='cubic', smooth=0)
    Z_rbf = rbf3(X, Y)

    #plt.figure(figsize=(14, 8))
    #plt.contourf(X, Y, Z, 1000, cmap="Blues")
    #plt.scatter(theta, cosphi, marker="x", color="black")
    #plt.savefig(args.out + "heatmap_mesh_test.png", bbox_inches='tight')

    # Extrapolation using a RBF with low smoothness
    mask = isnan(Z)
    z_min, z_max = min(valore), max(valore)
    idx = where(~mask, Z, Z_rbf)
    idx[idx < z_min] = 0
    idx[idx > z_max] = z_max

    # Plot interpolated+extrapoleted map
    path = os.path.join(args.out, f"heatmap.{basename}.{args.format}")
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(111)
    plt.contourf(X, Y, idx, n_countourf, cmap="coolwarm_r",
                 norm=Normalize(vmin=0., vmax=1.))
    #plt.scatter(theta, cosphi, marker="x", color="black")
    plt.xlabel("φ")
    # plt.ylabel("cos(θ)")
    plt.rcParams['font.size'] = 30
    #plt.xticks(color='w')
    plt.yticks(color='w')
    # Modifica dello spessore degli assi
    for axis in ['top', 'bottom', 'left', 'right']:
        ax.spines[axis].set_linewidth(3)
    plt.tick_params('x', width=3, length=14)
    plt.tick_params('y', width=3, length=14)
    plt.savefig(path, dpi=args.dpi, bbox_inches='tight')

    # Use scipy voronoi to construct the plot and use the matplotlib api
    if args.vorplot:
        path = os.path.join(args.out, f"voronoi.{basename}.{args.format}")
        vor_points = concatenate((x[:, None], y[:, None]), axis=1)
        vor = Voronoi(vor_points)
        fig, ax = plt.subplots(figsize=(14, 14))
        voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False,
                        line_colors="darkslategray", line_width=3)
        cs = ax.contourf(X, Y, idx, n_countourf, cmap="coolwarm_r",
                         norm=Normalize(vmin=0., vmax=1.))
        #ax.scatter(x, y, marker="o", color="darkslategray", s=100)
        ax.scatter(y, marker="o", color="darkslategray", s=100)
        plt.xlim([amin(X), amax(X)])
        plt.ylim([amin(Y), amax(Y)])
        plt.xlabel("φ")
        plt.ylabel("cos(θ)")
        plt.rcParams['font.size'] = 30
        plt.xticks(color='w')
        # plt.yticks(color='w')
        for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(3)    
        plt.savefig(path, dpi=args.dpi, bbox_inches='tight')

    # Dummy colorbar
    if args.bar:
        # Plot the colorbar after normalizing from 0 to 1
        def NormalizeData(data):
            return (data - amin(data)) / (amax(data) - amin(data))
        fig, ax = plt.subplots(figsize=(14, 14))
        cs = ax.contourf(X, Y, NormalizeData(idx), 100, cmap="coolwarm_r",
                         norm=Normalize(vmin=0., vmax=1.))
        ax.scatter(x, y, marker="o", color="darkslategray", s=100)
        plt.xlim([amin(X), amax(X)])
        plt.ylim([amin(Y), amax(Y)])
        # Colorbar ticks
        orientation="horizontal" #horizontal o vertical

        #Cambio dello spessore della colorbar
        import matplotlib as mpl
        mpl.rcParams['axes.linewidth'] = 3

        cbar = plt.colorbar(cs, ax=ax, orientation=orientation, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        if cbar.orientation == "vertical":
         cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
        else:
         cbar.ax.set_xticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
        cbar.ax.tick_params(size = 6, width = 3)
        path = os.path.join(args.out, f"colorbar.{basename}.{args.format}")
        plt.savefig(path, dpi=args.dpi, bbox_inches='tight')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Script to create heatmaps and voronoi maps from hydration shell data .txt .')
    parser.add_argument("-i", "--input", type=str, dest='inp', default=[],
                        nargs="+", help="Input path/paths to .txt data or to directories.")
    parser.add_argument("-o", "--output", type=str, dest='out', default=(os.path.abspath(os.path.dirname(sys.argv[0]))),
                        help="Output path for generated files.")
    parser.add_argument("-l", "--list", type=int, dest="col",
                        nargs="+", default=[5, 7, 8], help="Columns of the data")
    parser.add_argument("-n", "--number", type=int, dest='n',
                        help="Number of interpolations points and mesh",
                        default=50)
    parser.add_argument("-d", "--dpi", type=int, dest='dpi',
                        help="Dpi of the images.",
                        default=150)
    parser.add_argument("-vo", "--voronoi", type=bool, dest='vorplot',
                        help="Plot the voronoi tassellation for the data points.", default=False)
    parser.add_argument("-cbar", "--colorbar", type=bool, dest='bar',
                        help="Plot a dummy graph with the colorbar.", default=False)
    parser.add_argument("-f", "--file", type=str, dest='format',
                        help="File format to save images.", default="png")

    # parser.add_argument("-r", "--radius", type=int, dest='radius',
    #                    help="Radius of the spherical mesh", default=32)
    # parser.add_argument("-m", "--mesh", type=bool, dest='mesh',
    #                    help="Create mesh", default=False)
    # parser.add_argument("-v", "--visualize", type=bool, dest='visualize',
    #                    help="Flag to display the mesh", default=False)

    args = parser.parse_args()
    files = []
    for i, input_path in enumerate(args.inp):
        if os.path.isdir(input_path):
            for file in os.listdir(input_path):
                if file.endswith(".txt"):
                    files.append(os.path.join(input_path, file))
        else:
            files.append(input_path)

    for in_file in files:
        try:
            main(in_file, args)
        except OSError as error:
            print("WARNING:", error)

    """ Edited out as now it s not important
    if args.mesh:
        try:
            import pyvista as pv

        except ImportError as error:
            raise error

        # Construction of the spherical mesh

        xx, yy, zz = np.meshgrid(np.radians(np.linspace(0, 361, args.n+1)),
                                 np.radians(np.linspace(-90, 90, args.n+1)),
                                 [0])

        radius = args.radius
        x = radius * np.cos(yy) * np.cos(xx)
        y = radius * np.cos(yy) * np.sin(xx)
        z = radius * np.sin(yy)

        grid = pv.StructuredGrid(x, y, z)
        grid.cell_arrays['test'] = np.array(idx).ravel(order="F")
        grid.save(args.out + "mesh.vtk", binary=True)

        if args.visualize:
            p = pv.Plotter()
            p.add_mesh(grid, interpolate_before_map=True)
            p.link_views()
            p.camera_position = [(-500, -150, 150),
                                 (15.0, 0.0, 0.0),
                                 (0.00, 0.37, 0.93)]
            p.show()
    """
