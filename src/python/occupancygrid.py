'''
Defines the OccupancyGrid class which stores an occupancy grid and its
associated metadata.
Defines a mehtod that converts an occupancy grid into a set of points describing 
the center line, the inner edge, and the outer edge. These points use the global
coordinate frame.
'''

from PIL import Image
import numpy as np
import yaml
import matplotlib.pyplot as plt
from scipy import ndimage


class OccupancyGrid:
    def __init__(self, track):
        image = Image.open(f'tracks/{track}/{track}_map.png')
        array = np.array(image)
        if array.ndim != 2:
            raise Exception('Track is not grayscale')
        with open(f'tracks/{track}/{track}_map.yaml', 'r') as f:
            data = yaml.safe_load(f)
        threshold = (1 - data['occupied_thresh']) * 255
        if data['negate'] == 0:
            self.grid = array < threshold
        else:
            self.grid = array > threshold
        self.resolution = data['resolution']
        self.origin = data['origin']
        self.__get_edges()
    
    def __get_edges(self):
        regions, num_edges = ndimage.label(self.grid)
        if num_edges != 2:
            raise Exception("Failed to find two sets of edges")
        edges_1 = regions == 1
        edges_2 = regions == 2
        mid = self.grid.shape[1] // 2
        for row in range(self.grid.shape[0]):
            if self.grid[row][mid]:
                if edges_1[row][mid]:
                    self.outer_mask = edges_1
                    self.inner_mask = edges_2
                else:
                    self.outer_mask = edges_2
                    self.inner_mask = edges_1
                break
        else:
            raise Exception("Failed to identify outer from inner edge")
    
    def __get_edges_points(self):
        outer_mask = self.outer_mask
        inner_mask = self.inner_mask
        outer_points = []
        inner_points = []
        zeros = np.zeros((4, 4), dtype=np.uint8)
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                pass
    
    def __pixel_to_coords(self, row, col):
        y = row * self.resolution
        x = col * self.resolution
        dist = np.sqrt(x**2 + y**2)


    # shows the current occupancy grid as an image
    def view(self):
        array = self.grid * 255
        array = array.astype(np.uint8)
        plt.imshow(array, cmap='gray')
        plt.show()
    
    # shows the occupancy grid with more details
    def view_complete(self):
        array = self.outer_mask * 128 + self.inner_mask * 255
        array = array.astype(np.uint8)
        plt.imshow(array, cmap='gray')
        plt.show()

        

grid = OccupancyGrid('Austin')
grid.view_complete()