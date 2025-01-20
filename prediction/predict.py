# predict.py: predicts human agent behavior between two sightings

import math, numpy, cv2, heapq
from utils import dist_sq

# given a start and goal point, return the objects seen
def get_objects_visible_from_last_seen(start_location, end_location, map_boundaries, robot_dsg, human_fov, end_direction=None, debug_tag=""):
    """
    Given the start and end location of a human agent, predict the path and return the objects visible from the path.
    
    Args:
        start_location (list): The last seen location of the human agent as a continuous x/y location.
        end_location (list): The current location of the human agent as a continuous x/y location.
        map_boundaries (numpy.ndarray): The map boundaries image, (0,0,0,1) for wall, (0,0,0,0) for empty).
        robot_dsg (mental_model.MentalModel): The robot's mental model.
        human_fov (float): The human's field of view.
        end_direction (list, optional): The direction of the human agent at the end location as a continuous x/y vector (no need to normalize). Defaults to None.
        debug_tag (str, optional): A tag to add to the map debug image (in "prediction_maps/"). Defaults to "".
    
    Returns:
        list: The objects visible to the human agent.    
    """
    last = __project_continuous_location_to_map_location__(start_location, map_boundaries)  # get the last seen location of the human in pixel coordinates
    curr = __project_continuous_location_to_map_location__(end_location, map_boundaries)  # get the current location of the human in pixel coordinates
    path, all_neighbors = predict_path(last, curr, map_boundaries)  # [:2] to get only east and north (x and y)
    object_locations = []
    object_index_to_key = {}
    for i, o in enumerate(robot_dsg.objects):  # get the location of robot's known objects in pixel coordinates
        pos = __project_continuous_location_to_map_location__([robot_dsg.objects[o]["x"], robot_dsg.objects[o]["y"]], map_boundaries)
        object_locations.append([pos[0], pos[1], i])
        object_index_to_key[i] = o  # store the index of the object for referencing
    object_locations = numpy.array(object_locations)  # convert to numpy array for easier indexing
    # get the objects visible from the path
    visibility, visible_points = __get_objects_visible_from_path__(robot_dsg, path, map_boundaries, human_fov)  # visibility is [False, True, True, False, ...] for each object, visible_points is for debugging
    # if a direction is provided, get the objects visible from the last point 
    if end_direction is not None:
        visibility, visible_points = __get_objects_visible_from_point__(robot_dsg, curr, end_direction, map_boundaries, human_fov, object_locations=[], visibility=visibility, debug_visible_points=visible_points)
    visible_objects = object_locations[visibility]
    invisible_objects = object_locations[~numpy.array(visibility)]
    human_trajectory_debug = debug_path(robot_dsg, map_boundaries, path, all_neighbors, visible_objects, visible_points, invisible_objects, human_fov, tag=str(debug_tag))  # save a path png for debugging
    # organize the visible objects
    objects_visible_to_human = []
    for i, object_is_visible in enumerate(visibility):
        if object_is_visible:
            objects_visible_to_human.append(robot_dsg.objects[object_index_to_key[i]].as_dict()) 
    return objects_visible_to_human, human_trajectory_debug

def predict_path(previous_location, current_location, map_boundaries):
    # A* planner to predict the human agent's path
    # set inputs to normal lists
    if isinstance(previous_location, numpy.ndarray):
        previous_location = previous_location.tolist()
    if isinstance(current_location, numpy.ndarray):
        current_location = current_location.tolist()
    frontier = []
    heapq.heappush(frontier, (dist_sq(previous_location, current_location), previous_location))
    came_from = {}
    all_neighbors = []
    came_from[str(previous_location)] = (None, 0)
    while len(frontier) > 0:
        current = heapq.heappop(frontier)[1]
        if current == current_location:
            break
        for next in __neighbors__(current, map_boundaries):
            all_neighbors.append(next)
            if str(next) not in came_from:
                dist = dist_sq(next, current_location)
                heapq.heappush(frontier, (dist + came_from[str(current)][1], next))
                came_from[str(next)] = (current, dist + came_from[str(current)][1])
    path = []
    current = current_location
    while current != previous_location and str(current) in came_from:
        path.append(current)
        current = came_from[str(current)][0]
    path.append(previous_location)
    path.reverse()
    return path, all_neighbors

def __project_continuous_location_to_map_location__(continuous_location, map_boundaries) -> numpy.ndarray:
    # project the continuous location to the map location
    center = [-1, -4]
    dim = 23
    map_location = [int((continuous_location[0] - center[0] + dim/2) * map_boundaries.shape[0] / dim), int((continuous_location[1] - center[1] + dim/2) * map_boundaries.shape[1] / dim)]
    return numpy.array(map_location)

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

# get the objects visible from a path
def __get_objects_visible_from_path__(dsg, path, map_image, fov):
    # make sure path is a numpy array
    if not isinstance(path, numpy.ndarray):
        path = numpy.array(path)
    # check if the object is visible from the path
    visibility = [False for _ in range(len(dsg.objects))]
    debug_visible_points = []
    object_locations = numpy.array([[dsg.objects[obj]["x"], dsg.objects[obj]["y"], dsg.objects[obj]["z"]] for obj in dsg.objects])
    for i in range(1, len(path)):
        location = path[i]
        direction = (path[i] - path[i-1]) / numpy.linalg.norm(path[i] - path[i-1])
        visibility, debug_visible_points = __get_objects_visible_from_point__(dsg, location, direction, map_image, fov, object_locations, visibility, debug_visible_points)
    return visibility, numpy.array(debug_visible_points)

# get the objects visible from a location and direction
def __get_objects_visible_from_point__(dsg, location, direction, map_image, fov, object_locations=[], visibility=[], debug_visible_points=[]):
    # if object locations are not provided, get them from the dsg
    if len(object_locations) == 0:
        object_locations = numpy.array([[dsg.objects[obj]["x"], dsg.objects[obj]["y"], dsg.objects[obj]["z"]] for obj in dsg.objects])
    # if visibility is not provided, initialize it
    if len(visibility) == 0:
        visibility = [False for _ in range(len(dsg.objects))]
    # check if each object is visible from the location
    for object_idx, obj in enumerate(object_locations):
        if visibility[object_idx]:  # if the object is already visible, no need to check again
            continue
        object_is_visible, _visible_points = __object_is_visible_from_point__(location, direction, obj, map_image, fov)
        # print data type of the debug_visible_points numpy array
        if isinstance(debug_visible_points, list):
            debug_visible_points += _visible_points
        elif len(debug_visible_points) == 0:
            debug_visible_points = numpy.array(_visible_points)
        elif len(_visible_points) > 0:
            debug_visible_points = numpy.concatenate((debug_visible_points, _visible_points), axis=0)
        if object_is_visible:
            visibility[object_idx] = True
    return visibility, debug_visible_points

# helper function to check if a line collides with a boundary on a pixel level
def __raycast__(map_boundaries, start, end):
    # if floats are passed in, remap them to the boundaries
    if not isinstance(start[0], numpy.int64) or not isinstance(start[1], numpy.int64):
        start = __project_continuous_location_to_map_location__(start, map_boundaries)
    if not isinstance(end[0], numpy.int64) or not isinstance(end[1], numpy.int64):
        end = __project_continuous_location_to_map_location__(end, map_boundaries)
    visible_points = []
    # check if the object is visible from the start location
    diff = numpy.array([end[0] - start[0], end[1] - start[1]])
    diff_norm = numpy.linalg.norm(diff)
    if diff_norm != 0:
        direction = diff / diff_norm
    else:
        return False, visible_points
    for i in range(1, int(diff_norm)):
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

def __object_is_visible_from_point__(location, direction, obj_location, map_boundaries, fov):
    # if floats are passed in for the object location, remap them to the boundaries
    if not isinstance(obj_location[0], int) or not isinstance(obj_location[1], int):
        obj_location = __project_continuous_location_to_map_location__(obj_location, map_boundaries)
    # check if the object is visible from the path segment
    direction = direction / numpy.linalg.norm(direction)  # normalize the direction
    object_location_wrt_agent_location = numpy.array([obj_location[0] - location[0], obj_location[1] - location[1]])
    object_direction_wrt_agent_location = numpy.linalg.norm(object_location_wrt_agent_location)
    if object_direction_wrt_agent_location != 0:
        angle = math.acos(min(1.0, max(-1, numpy.dot(object_location_wrt_agent_location, direction) / (object_direction_wrt_agent_location * numpy.linalg.norm(direction)))))
    else:
        angle = 0

    # if the object is not in the field of view, it can't be visible
    if numpy.rad2deg(angle) > fov / 2:
        return False, []

    # check if the object is visible from the start location via a raycast
    visible, visible_points = __raycast__(map_boundaries, location, obj_location)
    if visible:
        return True, visible_points
    return False, visible_points

def debug_path(dsg, map_image, path, all_neighbors, visible_objects, visible_points, invisible_objects, fov, tag=""):
    if isinstance(path, list):
        path = numpy.array(path)
        all_neighbors = numpy.array(all_neighbors)
    map_2d = map_image
    map_image = numpy.dstack((map_image, map_image, map_image, numpy.ones(map_image.shape[:2], dtype=numpy.uint8)))
    map_image[map_2d == 0] = (255, 255, 255, 0)  # free space
    map_image[map_2d == 1] = (0, 0, 0, 255)  # walls/obstacles
    # map_image[all_neighbors[:,0], all_neighbors[:,1]] = (255, 100, 100, 50)  # neighbors seen in A*
    if visible_points.size > 0:
        map_image[visible_points[:,0], visible_points[:,1]] += numpy.array([100, 200, 100, 150], dtype=numpy.uint8)  # points checked for object visibility
    map_image[path[:,0], path[:,1]] = (255, 0, 0, 255)  # path
    map_image[path[0,0], path[0,1]] = (0, 255, 0, 255)  # start point
    map_image[path[-1,0], path[-1,1]] = (0, 0, 255, 255)  # goal point
    map_image[visible_objects[:,0], visible_objects[:,1]] = (255, 255, 0, 255)  # visible objects
    map_image[invisible_objects[:,0], invisible_objects[:,1]] = (100, 100, 0, 255)  # invisible objects
    scale_ratio = 10
    map_image = cv2.resize(map_image, None, fx=scale_ratio, fy=scale_ratio, interpolation=cv2.INTER_NEAREST)
    cv2.imwrite('prediction_maps/map_path' + ('_' + tag if tag != "" else '') + '.png', map_image)
    return map_image
