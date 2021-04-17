import argparse
import os
import sys
from numpy import linspace, array, loadtxt, meshgrid, isnan, where, concatenate, pi
from numpy import amin, amax
from scipy.interpolate import griddata, Rbf
from scipy.spatial import Voronoi, voronoi_plot_2d
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt


# Esempio di comando per eseguirlo:
# python map_vor_v0.1.py -i /Users/gab/Desktop/watermaps/solvent_accessibility.stride5_dist18.00_samples200.txt --vo true
# Path di output ora e` default dove si sta eseguendo il file!
# Ci sono varie flag, per vederle:
# python map_vor_v0.1.py --help
# Per eseguire lo script da spyder fare (CTRL + F6): >run>configuration per file
# In General Setting abilitare Command Line Option inserendo i vari comandi nel riquadro a destra dell'opzione
# Infine eseguire il file tramite run

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Script to create a spherical mesh from heatmap data.')
    parser.add_argument("-i", "--input", type=str, dest='inp',
                        help="Input path to .txt data")
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
                        help="Plot the voronoi tassellation for the points", default=False)
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

    basename = os.path.basename(args.inp).rsplit('.', 1)[
        0]  # Remove the file format
    t, c, v = (array(args.col)-1)
    array = loadtxt(args.inp)
    theta, cosphi, valore = array[:, t], array[:, c], array[:, v]

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
    plt.figure(figsize=(14, 14))
    plt.contourf(X, Y, idx, 700, cmap="Blues")
    #plt.scatter(theta, cosphi, marker="x", color="black")
    plt.savefig(path, dpi=args.dpi, bbox_inches='tight')

    # Use scipy voronoi to construct the plot and use the matplotlib api
    if args.vorplot:
        path = os.path.join(args.out, f"voronoi.{basename}.{args.format}")
        vor_points = concatenate((x[:, None], y[:, None]), axis=1)
        vor = Voronoi(vor_points)
        fig, ax = plt.subplots(figsize=(14, 14))
        voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False,
                        line_colors="darkslategray", line_width=3)
        cs = ax.contourf(X, Y, idx, 700, cmap="Blues",
                         norm=Normalize(vmin=0., vmax=1.))
        ax.scatter(x, y, marker="o", color="darkslategray", s=100)
        plt.xlim([amin(X), amax(X)])
        plt.ylim([amin(Y), amax(Y)])
        plt.xlabel("φ")
        plt.ylabel("cos(θ)")
        plt.savefig(path, dpi=args.dpi, bbox_inches='tight')

    # Dummy colorbar
    if args.bar:
        # Plot the colorbar after normalizing from 0 to 1
        def NormalizeData(data):
            return (data - amin(data)) / (amax(data) - amin(data))
        fig, ax = plt.subplots(figsize=(14, 14))
        cs = ax.contourf(X, Y, NormalizeData(idx), 100, cmap="Blues",
                         norm=Normalize(vmin=0., vmax=1.))
        ax.scatter(x, y, marker="o", color="darkslategray", s=100)
        plt.xlim([amin(X), amax(X)])
        plt.ylim([amin(Y), amax(Y)])
        # Colorbar ticks
        cbar = plt.colorbar(cs, ax=ax, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        cbar.ax.set_yticklabels(['0', '0.2', '0.4', '0.6', '0.8', '1'])
        path = os.path.join(args.out, f"colorbar.{basename}.{args.format}")
        plt.savefig(path, dpi=args.dpi, bbox_inches='tight')

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
