import math

import torch
from csv import DictReader
from typing import Tuple

import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point


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
    def __init__(self, boundary_file_name: str, cell_width: float):
        """
        :param boundary_file_name: filename of a csv with lat,lng of the boundary
        :param cell_width:
        """

        self.cells = []
        self.cell_width = cell_width
        self.boundary = Boundary(boundary_file_name)

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

    def plot(self):
        self.boundary.plot()

        for cell in self.cells:
            x = [v[0] for v in cell.exterior.coords]
            y = [v[1] for v in cell.exterior.coords]

            plt.plot(x, y)

    def one_hot(self, coordinate: Tuple[float, float]):
        coordinate_point = Point(coordinate[::-1])

        return torch.Tensor([1.0 if cell.contains(coordinate_point) else 0.0 for cell in self.cells])
