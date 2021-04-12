
import os
import numpy as np
from scipy.interpolate import griddata, Rbf
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.spatial import Voronoi, voronoi_plot_2d, KDTree

# r"C:\Users\gabcr\Desktop\Uni_research\watermaps\solvent_accessibility.stride5_dist10.00_samples200.txt"
# example  -i C:\Users\gabcr\Desktop\Uni_research\watermaps\solvent_accessibility.stride5_dist10.00_samples200.txt -n 300 -v true -o C:\Users\gabcr\Desktop\

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Script to create a spherical mesh from heatmap data.')
    parser.add_argument("-i", "--input", type=str, dest='inp',
                        help="Input path to .txt data")
    parser.add_argument("-o", "--output", type=str, dest='out',
                        help="Output path for files")
    parser.add_argument("-n", "--number", type=int, dest='n',
                        help="Number of interpolations points and mesh",
                        default=50)
    parser.add_argument("-vo", "--voronoi", type=bool, dest='vorplot',
                        help="Plot the voronoi tassellation for the points", default=False)
    parser.add_argument("-r", "--radius", type=int, dest='radius',
                        help="Radius of the spherical mesh", default=32)
    parser.add_argument("-l", "--list", type=int, dest="col",
                        nargs="+", default=[5, 7, 8], help="Columns of the data")
    parser.add_argument("-m", "--mesh", type=bool, dest='mesh',
                        help="Create mesh", default=False)
    parser.add_argument("-v", "--visualize", type=bool, dest='visualize',
                        help="Flag to display the mesh", default=False)
    args = parser.parse_args()

    t, c, v = (np.array(args.col)-1)
    array = np.loadtxt(args.inp)
    theta, cosphi, valore = array[:, t], array[:, c], array[:, v]

    # Interpolation of the data points to construct the 2D heatmap

    x, y, z = theta, cosphi, valore
    xi = np.linspace(-np.pi, np.pi, args.n)
    yi = np.linspace(-1, 1, args.n)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata((x, y), z, (X, Y), method='cubic', rescale="true")
    rbf3 = Rbf(x, y, z, function='cubic', smooth=0)
    Z_rbf = rbf3(X, Y)

    plt.figure(figsize=(14, 8))
    plt.contourf(X, Y, Z, 1000, cmap="jet_r")
    plt.scatter(theta, cosphi, marker="x", color="black")
    plt.savefig(args.out + "heatmap_mesh_test.png")

    mask = np.isnan(Z)
    z_min, z_max = min(valore), max(valore)
    idx = np.where(~mask, Z, Z_rbf)
    idx[idx < z_min] = 0
    idx[idx > z_max] = z_max

    # Plot interpolated+extrapoleted map
    plt.figure(figsize=(14, 8))
    plt.contourf(X, Y, idx, 1000, cmap="Blues")
    #plt.scatter(theta, cosphi, marker="x", color="black")
    plt.savefig(args.out + "heatmap_mesh.png")

    if args.vorplot:
        vor_points=np.concatenate((x[:,None],y[:,None]), axis=1)
        vor = Voronoi(vor_points)
        fig, ax = plt.subplots(figsize=(14,8))
        voronoi_plot_2d(vor, ax=ax, show_points=False, show_vertices=False)
        ax.contourf(X, Y, idx, 1000, cmap="Blues")
        ax.scatter(x,y,marker="o", color="black")
        plt.xlim([np.min(X),np.max(X)])
        plt.ylim([np.min(Y),np.max(Y)])
        plt.savefig(args.out+"voronoi_plot.png", dpi=200)


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
