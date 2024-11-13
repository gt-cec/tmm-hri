import math, numpy

location = [.473466,-5.844573,0.973318]
direction = [0,-1,0]
object_loc = [2.81,-10.81,0.95]
object_location_wrt_agent_location = numpy.array([object_loc[0] - location[0], object_loc[1] - location[1], object_loc[2] - location[2]])
# check if object is in the agent's field of view
# NOTE: will need to filter by walls/visibility
# direction = direction / numpy.linalg.norm(direction)
print(">>>", direction)
angle = math.acos(numpy.dot(object_location_wrt_agent_location, direction) / (numpy.linalg.norm(object_location_wrt_agent_location) * numpy.linalg.norm(direction)))

print("Angle", numpy.rad2deg(angle))
