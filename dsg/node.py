# node.py: node class for environment objects

import time  # for getting the time seen
import math

# the Node class is an object in the scene
class Node:
    def __init__(self, object_class=None, x=None, y=None, z=None, last_seen=None):
        self.object_class = object_class
        self.x, self.y, self.z = x, y, z
        self.last_seen = last_seen if last_seen is not None else int(time.time())  # default the last_seen to the current time
        self.updates = 0
        return

    # updates a property of the node
    def update(self, object_class=None, x=None, y=None, z=None, last_seen=None):
        if object_class is not None:
            self.object_class = object_class
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z
        self.last_seen = last_seen if last_seen is not None else int(time.time())
        self.updates += 1
        return

    # computes the euclidean distance to a given coordinate
    def distance(self, x=None, y=None, z=None):
        if x is None or y is None or z is None:  # check if the target has a null location
            raise ValueError("Could not calculate the distance because the target location (" + str(x) + ", " + str(y) + ", " + str(z) + ") has a null value")
        if self.x is None or self.y is None or self.z is None:  # check if this object has a null location
                raise ValueError("Could not calculate the distance because this object's location (" + str(self.x) + ", " + str(self.y) + ", " + str(self.z) + ") has a null value")
        return math.sqrt((x - self.x) ** 2 + (y - self.y) ** 2 + (z - self.z) ** 2)  # return the euclidean distance

    # allows the Node to be cast as a dictionary
    def __iter__(self):
        yield ("class", self.object_class)
        yield ("x", self.x)
        yield ("y", self.y)
        yield ("z", self.z)
        yield ("last seen", self.last_seen)
        yield ("updates", self.updates)
        return

    # returns a dictionary representation of the Node
    def as_dict(self) -> dict:
        return {
            "class": self.object_class,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "last seen": self.last_seen,
            "updates": self.updates
        }

    # returns a string representation of the node
    def __str__(self) -> str:
        return str(self.as_dict())

    # returns the object with the given key
    def __getitem__(self, key):
        if key == "class":
            return self.object_class
        elif key == "x":
            return self.x
        elif key == "y":
            return self.y
        elif key == "z":
            return self.z
        elif key == "last seen":
            return self.last_seen
        elif key == "updates":
            return self.updates
        else:
            raise KeyError("The given key '" + str(key) + "' is not an object property, valid properties are: class, x, y, z, last seen, updates.")
