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
from skimage.morphology import skeletonize


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
        self.get_centerline()
        self.__get_edges_points()
        self.get_centerline_points()
        self.__align_phases()
        self.__align_frequencies()
    
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
        self.outer_points_raw = self.points_from_skeleton(skeletonize(self.outer_mask), dist=5)
        self.inner_points_raw = self.points_from_skeleton(skeletonize(self.inner_mask), dist=5)
        return
        outer_mask = np.copy(self.outer_mask)
        inner_mask = np.copy(self.inner_mask)
        outer_points = []
        inner_points = []
        rad = 3
        zeros = np.zeros((5, 5), dtype=np.uint8)
        for row in range(self.grid.shape[0]):
            for col in range(self.grid.shape[1]):
                if inner_mask[row][col]:
                    inner_points.append([col * self.resolution, row * self.resolution])
                    inner_mask[(row - rad):(row + rad + 1), (col - rad):(col + rad + 1)] = False
                if outer_mask[row][col]:
                    outer_points.append([col * self.resolution, row * self.resolution])
                    outer_mask[(row - rad):(row + rad + 1), (col - rad):(col + rad + 1)] = False
        self.outer_points_raw = np.array(outer_points, dtype=np.float32)
        self.inner_points_raw = np.array(inner_points, dtype=np.float32)

    def get_centerline(self, res=1, plot=False):
        # upscale edges by res
        outer_mask_tmp = np.repeat(self.outer_mask, res, axis=1)
        inner_mask_tmp = np.repeat(self.inner_mask, res, axis=1)
        outer_mask = np.repeat(outer_mask_tmp, res, axis=0)
        inner_mask = np.repeat(inner_mask_tmp, res, axis=0)
        # mask to dilate by
        mask = np.array([
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ])
        intersection_prev = inner_mask & outer_mask
        intersection = inner_mask & outer_mask
        iteration = 0
        iteration_without_change = 0
        while not np.array_equal(intersection, intersection_prev) or np.sum(intersection) == 0 or iteration_without_change < 10:
            if plot and iteration % 10 == 0:
                fig, axes = plt.subplots(1, 3)
                axes[0].imshow((inner_mask * 255).astype(np.uint8), cmap='gray')
                axes[1].imshow((outer_mask * 255).astype(np.uint8), cmap='gray')
                axes[2].imshow(OccupancyGrid.__dilate_copy(intersection).astype(np.uint8), cmap='gray')
                plt.show()
            inner_mask = ndimage.binary_dilation(inner_mask, structure=mask)
            outer_mask = ndimage.binary_dilation(outer_mask, structure=mask)
            intersection_prev = intersection
            intersection = np.copy(intersection) | inner_mask & outer_mask
            inner_mask &= ~intersection
            outer_mask &= ~intersection
            iteration += 1
            # print(iteration_without_change)
            print(np.sum(intersection) - np.sum(intersection_prev))
            print(iteration_without_change)
            if np.array_equal(intersection_prev, intersection):
                iteration_without_change += 1
            else:
                iteration_without_change = 0
        self.intersection_mask = intersection

    def get_centerline_points(self):
        self.intersection_points_raw = self.points_from_skeleton(skeletonize(self.intersection_mask), dist=5)

    def points_from_skeleton(self, skeleton, dist=4):
        points = set()
        edges = dict()
        num_rows, num_cols = skeleton.shape[0], skeleton.shape[1]
        rows, cols = np.where(skeleton)
        for y, x in zip(rows, cols):
                points.add((y, x))
                edges[(y, x)] = list()
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < num_rows and 0 <= nx < num_cols and skeleton[ny, nx]:
                            edges[(y, x)].append((ny, nx))
        visited_points = set()
        ordered_points = list()
        point = points.pop()
        points.add(point)
        while point not in visited_points:
            visited_points.add(point)
            ordered_points.append(point)
            if len(edges[point]) != 2:
                print(f"point without 2 edges: {point}. Has {len(edges[point])}")
            for neighbour in edges[point]:
                if neighbour not in visited_points:
                    point = neighbour
                    break
        sampled_points = [ordered_points[0]]
        prev_y, prev_x = ordered_points[0][0], ordered_points[0][1]
        for y, x in ordered_points[1:]:
            dist_squared = (y - prev_y) ** 2 + (x - prev_x) ** 2
            if dist_squared >= dist * dist:
                sampled_points.append((y, x))
                prev_y, prev_x = y, x

        # currently as pixel coordinates (row, col), but real world are (x, y)
        np_points = np.asarray(sampled_points, dtype=np.uint32)
        np_points[:, [0, 1]] = np_points[:, [1, 0]]
        return np_points * self.resolution    

    def __align_phases(self):
        origin = self.intersection_points_raw[0]
        diff_inner = self.inner_points_raw - origin
        diff_outer = self.outer_points_raw - origin
        dist_inner = np.linalg.norm(diff_inner, axis=1)
        dist_outer = np.linalg.norm(diff_outer, axis=1)
        start_inner = np.argmin(dist_inner)
        start_outer = np.argmin(dist_outer)
        self.inner_points_raw = np.roll(self.inner_points_raw, -start_inner, axis=0)
        self.outer_points_raw = np.roll(self.outer_points_raw, -start_outer, axis=0)
        # align orientation (i.e. all clockwise or counterclockwise)
        next_center = self.intersection_points_raw[1]
        prev_center = self.intersection_points_raw[-1]
        next_inner = self.inner_points_raw[1]
        prev_inner = self.inner_points_raw[-1]
        next_outer = self.outer_points_raw[1]
        prev_outer = self.outer_points_raw[-1]
        if np.linalg.norm(next_center - next_inner) > np.linalg.norm(next_center - prev_inner):
            self.inner_points_raw = np.flip(self.inner_points_raw, axis=0)
        if np.linalg.norm(next_center - next_outer) > np.linalg.norm(next_center - prev_outer):
            self.outer_points_raw = np.flip(self.outer_points_raw, axis=0)

    def __align_frequencies(self, sampling_points=-1):
        if sampling_points <= 1:
            sampling_points = self.intersection_points_raw.shape[0]
        arc_intersection = OccupancyGrid.__arclength(self.intersection_points_raw)
        arc_inner = OccupancyGrid.__arclength(self.inner_points_raw)
        arc_outer = OccupancyGrid.__arclength(self.outer_points_raw)
        intersection = np.empty((sampling_points, 2), dtype=np.float32)
        inner = np.empty((sampling_points, 2), dtype=np.float32)
        outer = np.empty((sampling_points, 2), dtype=np.float32)
        for i in range(sampling_points):
            t = i / sampling_points
            intersection[i] = OccupancyGrid.__query_arclength(self.intersection_points_raw, arc_intersection, t)
            inner[i] = OccupancyGrid.__query_arclength(self.inner_points_raw, arc_inner, t)
            outer[i] = OccupancyGrid.__query_arclength(self.outer_points_raw, arc_outer, t)
            # inner[i] = OccupancyGrid.__query_arclength_project(self.inner_points_raw, arc_inner, t, intersection[i])
            # outer[i] = OccupancyGrid.__query_arclength_project(self.outer_points_raw, arc_outer, t, intersection[i])
        self.intersection_points = intersection
        self.inner_points = inner
        self.outer_points = outer

    def __arclength(arr):
        diff = np.diff(arr, axis=0)
        norms = np.linalg.norm(diff, axis=1)
        dists = np.concatenate(([0], np.cumsum(norms)))
        dists /= dists[-1]
        return dists
    
    def __query_arclength(arr, lengths, t):
        idx = np.searchsorted(lengths, t) - 1
        left = t - lengths[idx]
        if -0.0001 < left < 0.0001:
            return arr[idx]
        right = lengths[idx + 1] - t
        normalization = 1 / (left + right)
        left, right = left * normalization, right * normalization
        return arr[idx] * (1 - left) + arr[idx+1] * (1 - right)
    
    def __query_arclength_project(arr, lengths, t, point):
        idx = np.searchsorted(lengths, t) - 1
        in_range, projection = OccupancyGrid.__project_if_in_range(arr[idx], arr[idx+1], point)
        if in_range < 0:
            in_range, projection = OccupancyGrid.__project_if_in_range(arr[idx-1], arr[idx], point)
            if in_range < 0:
                print("coefficient too small")
            if in_range > 0:
                print("should never happen")
        elif in_range > 1:
            in_range, projection = OccupancyGrid.__project_if_in_range(arr[idx+1], arr[idx+2], point)
            if in_range > 0:
                print("coefficient too large")
            if in_range < 0:
                print("should never happen")
        return projection
        v = arr[idx+1] - arr[idx]
        u = point - arr[idx]
        start = arr[idx]
        coefficient, projection = OccupancyGrid.__project(v, u)
        if coefficient < 0:
            v = arr[idx] - arr[idx-1]
            u = point - arr
            print(f"coefficient too small: {coefficient}")
            return arr[idx]
        if coefficient > lengths[idx + 1] - lengths[idx]:
            print(f"coefficient too large: {coefficient}")
            return arr[idx+1]
        return start + projection
    
    def __project_if_in_range(p1, p2, q):
        v = p2 - p1
        u = q - p1
        coefficient = np.dot(v, u) / np.dot(v, v)
        if coefficient < 0:
            return -1, p1
        if coefficient > np.linalg.norm(v):
            return +1, p2
        return 0, p1 + v * coefficient
        

    def view_centerline_mask(self):
        plt.imshow(skeletonize(self.intersection_mask), cmap='gray')
        plt.show()

    def __dilate_copy(arr):
        mask = np.ones((7, 7), dtype=np.bool)
        copy = np.copy(arr)
        return ndimage.binary_dilation(copy, structure=mask)


    # shows the current occupancy grid as an image
    def view(self):
        array = self.grid * 255
        array = array.astype(np.uint8)
        plt.imshow(array, cmap='gray')
        plt.show()
    
    # shows the occupancy grid with more details
    def view_complete(self):
        # edge masks
        # array = self.outer_mask * 128 + self.inner_mask * 255
        # array = array.astype(np.uint8)
        # plt.imshow(array, cmap='gray')
        # plt.show()
        # edges as points
        plt.scatter(*(self.inner_points[0 : 500, :]).T, c='red', s=0.1, alpha=1)
        plt.scatter(*(self.outer_points[0 : 500, :]).T, c='blue', s=0.1, alpha=1)
        plt.scatter(*(self.intersection_points[0 : 500, :]).T, c='black', s=0.1, alpha=1)
        plt.show()

        
if __name__ == '__main__':
    grid = OccupancyGrid('Austin')
    grid.view_complete()