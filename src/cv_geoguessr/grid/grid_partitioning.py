import math

import random
import torch
from csv import DictReader
from random import uniform
from typing import Tuple

import matplotlib.pyplot as plt

from scipy.spatial import Voronoi, voronoi_plot_2d
import numpy as np
from shapely.geometry import Polygon, Point
import shapely
import shapely.ops

def parse_boundary_csv(file_name: str):
    """
    Reads and parses a csv file with city boundaries
    :param file_name: filepath of the csv file
    :return: dict with 'verts' a list of tuples (lng, lat)
        'min_lat', 'max_lat', 'min_lng', 'max_lng' the min and max
    """

    boundary = {
        'verts': [],
        'min_lat': None,
        'max_lat': None,
        'min_lng': None,
        'max_lng': None
    }

    with open(file_name) as csv_file:
        reader = DictReader(csv_file)
        for row in reader:
            lat = float(row['lat'])
            lng = float(row['lng'])

            # Add the vertex to the vertices list
            boundary['verts'].append(
                (lng, lat)
            )

            # Keep track of the min/max latitude
            if boundary['min_lat'] is None or boundary['min_lat'] > lat:
                boundary['min_lat'] = lat
            if boundary['max_lat'] is None or boundary['max_lat'] < lat:
                boundary['max_lat'] = lat

            # Keep track of the min/max longitude
            if boundary['min_lng'] is None or boundary['min_lng'] > lng:
                boundary['min_lng'] = lng
            if boundary['max_lng'] is None or boundary['max_lng'] < lng:
                boundary['max_lng'] = lng

    return boundary


class Boundary:
    def __init__(self, boundary_file_name: str):
        boundary = parse_boundary_csv(boundary_file_name)
        self.verts = boundary["verts"]
        self.min_lat = boundary['min_lat']
        self.max_lat = boundary['max_lat']
        self.min_lng = boundary['min_lng']
        self.max_lng = boundary['max_lng']
        self.width = self.max_lng - self.min_lng
        self.height = self.max_lat - self.min_lat
        self.center = (self.min_lng + 0.5 * self.width,
                       self.min_lat + 0.5 * self.height)
        self.polygon = Polygon(self.verts)

    def intersects_cube(self, top_left: Tuple[float, float], width: float):
        bottom_right = (top_left[0] - width, top_left[1] - width)

        return self.polygon.intersects([
            (top_left[0], top_left[1]),
            (top_left[0], bottom_right[1]),
            (bottom_right[0], bottom_right[1]),
            (bottom_right[0], top_left[1]),
        ])

    def intersects(self, other: Polygon):
        return self.polygon.intersects(other)

    def plot(self):
        x = [vert[0] for vert in self.verts]
        y = [vert[1] for vert in self.verts]
        x.append(self.verts[0][0])
        y.append(self.verts[0][1])
        plt.plot(x, y)
        # Plot bounds
        plt.plot(
            [self.min_lng, self.max_lng, self.max_lng, self.min_lng, self.min_lng],
            [self.max_lat, self.max_lat, self.min_lat, self.min_lat, self.max_lat]
        )
        plt.scatter(self.center[0], self.center[1])


class Partitioning:
    def __init__(self, boundary_file_name: str, cell_width: float, voronoi :bool = False):
        """
        :param boundary_file_name: filename of a csv with lat,lng of the boundary
        """

        self.cells = []
        self.cell_centers = []
        self.voronoi = voronoi
        self.boundary = Boundary(boundary_file_name)

        if voronoi:
            self.add_voronoi_cells(cell_width)
        else:
            self.add_square_cells(cell_width)

    def add_voronoi_cells(self, cell_width):
        # https://stackoverflow.com/questions/27548363/from-voronoi-tessellation-to-shapely-polygons

        # points = np.random.default_rng().uniform(size=(1000, 2),
        #                                          low=(self.boundary.min_lng, self.boundary.min_lat),
        #                                          high=(self.boundary.max_lng, self.boundary.max_lat))

        # outside = []
        # for i, point in enumerate(points):
        #     coordinate_point = Point(point)
        #     dist = self.boundary.polygon.distance(coordinate_point)
        #     # if not self.boundary.polygon.contains(coordinate_point):
        #     if dist > cell_width/2:
        #         outside.append(i)
        #         # plt.scatter(point[0], point[1])
        i = 0
        points = []
        random.seed(10)

        while i < 100:
            point = (uniform(self.boundary.min_lng, self.boundary.max_lng),
                     uniform(self.boundary.min_lat, self.boundary.max_lat))

            coordinate_point = Point(point)

            # skip points not in the boundary
            if not self.boundary.polygon.contains(coordinate_point):
                continue

            # get the minimum distance
            dist = self.boundary.polygon.exterior.distance(coordinate_point) * 2
            for other in points:
                other_dist = ((other[0]-point[0])**2 + (other[1] - point[1])**2)**0.5
                # print(other_dist)
                # print(dist)
                dist = min(dist, other_dist)

            if 0.5 * cell_width < dist < 0.7 * cell_width:
                points.append(point)
                self.cell_centers.append(point)
                # print(points)
                i = 0
            elif len(points) > self.boundary.width/cell_width:
                i += 1

        # Additional points to close off the structure
        n = 50
        t = 5
        top = self.boundary.max_lat + t * self.boundary.height
        bottom = self.boundary.min_lat - t * self.boundary.height
        left = self.boundary.min_lng - t * self.boundary.width
        right = self.boundary.max_lng + t * self.boundary.width
        for i in range(n+1):
            x = left + i/n * (2*t + 1) * self.boundary.width
            points.append((x, top))
            points.append((x, bottom))
        for i in range(n+1):
            y = bottom + i/n * (2*t + 1) * self.boundary.height
            points.append((left, y))
            points.append((right, y))

        vor = Voronoi(points)
        # voronoi_plot_2d(vor) # uncomment to plot voronoi

        lines = [
            shapely.geometry.LineString(vor.vertices[line])
            for line in vor.ridge_vertices
            if -1 not in line
        ]
        for poly in shapely.ops.polygonize(lines):
            if self.boundary.intersects(poly):
                intersection_poly = poly.intersection(self.boundary.polygon)
                self.cells.append(intersection_poly)
                # print(intersection_poly.area)



    def add_square_cells(self, cell_width):
        n_cells_x = math.ceil(self.boundary.width / cell_width)
        n_cells_y = math.ceil(self.boundary.height / cell_width)

        residual_x = n_cells_x * cell_width - self.boundary.width
        residual_y = n_cells_y * cell_width - self.boundary.height

        for j in range(n_cells_y):
            for i in range(n_cells_x):
                bottom = self.boundary.min_lat - 0.5 * residual_y + j * cell_width
                top = bottom + cell_width
                left = self.boundary.min_lng - 0.5 * residual_x + i * cell_width
                right = left + cell_width

                poly = Polygon([
                    (left, top),
                    (right, top),
                    (right, bottom),
                    (left, bottom),
                ])

                if self.boundary.intersects(poly):
                    self.cells.append(poly)

    def __len__(self):
        return len(self.cells)

    def plot(self):
        self.boundary.plot()

        for cell in self.cells:
            x = [v[0] for v in cell.exterior.coords]
            y = [v[1] for v in cell.exterior.coords]

            plt.plot(x, y)
            plt.scatter(cell.centroid.x, cell.centroid.y)

    def one_hot(self, coordinate: Tuple[float, float]):
        coordinate_point = Point(coordinate[::-1])

        return torch.Tensor([1.0 if cell.contains(coordinate_point) else 0.0 for cell in self.cells])
