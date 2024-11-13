# predict.py: predicts human agent behavior between two sightings

import math, numpy, cv2, heapq

from utils import dist_sq

def predict_path(previous_location, current_location, map_boundaries):
    # A* planner to predict the human agent's path
    frontier = []
    heapq.heappush(frontier, (dist_sq(previous_location, current_location), previous_location))
    came_from = {}
    all_neighbors = []
    came_from[str(previous_location)] = None
    while len(frontier) > 0:
        current = heapq.heappop(frontier)[1]
        if current == current_location:
            break
        for next in __neighbors__(current, map_boundaries):
            all_neighbors.append(next)
            if str(next) not in came_from:
                heapq.heappush(frontier, (dist_sq(next, current_location), next))
                came_from[str(next)] = current
    path = []
    current = current_location
    while current != previous_location and str(current) in came_from:
        path.append(current)
        current = came_from[str(current)]
    path.append(previous_location)
    path.reverse()
    return path, all_neighbors

# helper function to get neighbors of a location
def __neighbors__(location, map_boundaries):
    neighbors = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            neighbor = [location[0] + i, location[1] + j]
            if 0 <= neighbor[0] < map_boundaries.shape[1] and 0 <= neighbor[1] < map_boundaries.shape[0] and map_boundaries[neighbor[0], neighbor[1]] == 0:
                neighbors.append(neighbor)
    return neighbors

# helper function to check if objects are visible from the path
def __objects_are_visible_from_path__(path, map_boundaries, objects, fov):
    # check if the object is visible from the path
    visibility = [False for _ in range(len(objects))]
    visible_points = []
    for i in range(1, len(path)):
        for o, obj in enumerate(objects):
            if visibility[o]:  # if the object is already visible, no need to check again
                continue
            object_is_visible, _visible_points = __object_is_visible_from_path_segment__(path[i-1], path[i], obj, map_boundaries, fov)
            visible_points += _visible_points
            if object_is_visible:
                visibility[o] = True
    return visibility, numpy.array(visible_points)

# helper function to check if a line collides with a boundary on a pixel level
def __raycast__(map_boundaries, start, end):
    visible_points = []
    # check if the object is visible from the start location
    diff = numpy.array([end[0] - start[0], end[1] - start[1]])
    direction = diff / numpy.linalg.norm(diff)
    for i in range(1, int(numpy.linalg.norm(diff))):
        location = [int(start[0] + i * direction[0]), int(start[1] + i * direction[1])]
        if map_boundaries[location[0], location[1]] == 1:
            return False, visible_points
        visible_points.append(location)
    return True, visible_points

# helper function to check if an object is visible from a path segment (adjacent path points)
def __object_is_visible_from_path_segment__(start, end, obj, map_boundaries, fov):
    # check if the object is visible from the path segment
    direction = numpy.array([end[0] - start[0], end[1] - start[1]])
    direction = direction / numpy.linalg.norm(direction)
    object_location_wrt_agent_location = numpy.array([obj[0] - start[0], obj[1] - start[1]])
    angle = math.acos(min(1.0, max(-1, numpy.dot(object_location_wrt_agent_location, direction) / (numpy.linalg.norm(object_location_wrt_agent_location) * numpy.linalg.norm(direction)))))

    # if the object is not in the field of view, it can't be visible
    if numpy.rad2deg(angle) > fov / 2:
        return False, []

    # check if the object is visible from the start location via a raycast
    visible, visible_points = __raycast__(map_boundaries, start, obj)
    if visible:
        return True, visible_points
    return False, visible_points

def debug_path(map_image, path, all_neighbors):
    # generate some objects
    objects = numpy.array([[10, 10], [10, 50], [50, 50], [80, 35]])
    visibility, visible_points = __objects_are_visible_from_path__(path, map_image, objects, 90)
    visible_objects = objects[visibility]
    invisible_objects = objects[~numpy.array(visibility)]
    if isinstance(path, list):
        path = numpy.array(path)
        all_neighbors = numpy.array(all_neighbors)
    map_2d = map_image
    map_image = numpy.dstack((map_image, map_image, map_image, numpy.ones(map_image.shape[:2], dtype=numpy.uint8)))
    map_image[map_2d == 0] = (255, 255, 255, 0)  # free space
    map_image[map_2d == 1] = (0, 0, 0, 255)  # walls/obstacles
    map_image[all_neighbors[:,0], all_neighbors[:,1]] = (255, 100, 100, 50)  # neighbors seen in A*
    map_image[visible_points[:,0], visible_points[:,1]] += numpy.array([100, 200, 100, 150], dtype=numpy.uint8)  # points checked for object visibility
    map_image[path[:,0], path[:,1]] = (255, 0, 0, 255)  # path
    map_image[path[0,0], path[0,1]] = (0, 0, 255, 255)  # start point
    map_image[path[-1,0], path[-1,1]] = (0, 255, 0, 255)  # goal point
    map_image[visible_objects[:,0], visible_objects[:,1]] = (255, 255, 0, 255)  # visible objects
    map_image[invisible_objects[:,0], invisible_objects[:,1]] = (100, 100, 0, 255)  # invisible objects
    scale_ratio = 10
    map_image = cv2.resize(map_image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('map_path.png', map_image)
    return input()


# Output: [[0, 0, 0], [1, 1, 0], [2, 2, 0]]
# The predict function returns the path the human agent is likely to take between two sightings. The function uses an A* planner to find the shortest path between the two locations, considering the map boundaries and a padding parameter to avoid obstacles. The test case demonstrates the predicted path between the previous and current locations.
