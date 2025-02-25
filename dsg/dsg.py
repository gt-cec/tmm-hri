# dsg.py: constructs and maintains the scene graph

from .node import Node  # for scene object
from typing import Optional  # for arguments that can be None

# constructs, maintains, and resolves a scene graph from inputs of sighted object classes and locations
class DSG:
    def __init__(self, objects:list=[], same_location_threshold:int=4):
        # public variables
        self.objects = {}  # dictionary of object ID : Node (environment object)
        self.lost_object_ids = []  # list of object IDs that the agent has lost track of
        self.same_location_threshold = same_location_threshold  # distance threshold in which we consider an object unmoved

        # private variables
        self.__objects_by_class__ = {}  # cache the objects in each class, dictionary of object class : str (object ID)

        # if an objects list was provided, initialize the scene to those objects
        if objects != []:
            self.initialize_scene(objects)
        return

    # initializes the DSG from a list of objects
    def initialize_scene(self, objects:list, verbose=False) -> None:
        assert isinstance(objects, list), "Object list used to initialize the DSG must be a list of dictionaries, it is currently not a list"  # check that objects is a list
        assert len([True for x in objects if not isinstance(x, dict)]) == 0 , "Object list used to initialize the DSG must be a list of dictionaries, a list was passed but some values are not dictionaries" # check that objects only contains dictionaries
        self.__reset__()  # reset the DSG, clears all objects
        for obj in objects:  # add each object to the DSG
            assert "class" in obj, f"object is missing a 'class' attribute: {obj}"
            assert "x" in obj, f"object is missing a 'x' attribute: {obj}"
            assert "y" in obj, f"object is missing a 'y' attribute: {obj}"
            assert "z" in obj, f"object is missing a 'z' attribute: {obj}"
            self.add_object(obj)  # add the object
            if verbose:
                print("Initialized DSG with a new object:", obj)
        return

    # add a new object to the scene graph
    def add_object(self, object_dict=None, object_class=None, x=None, y=None, z=None):
        object_class, x, y, z = self.__validate_object_properties__(object_dict=object_dict, object_class=object_class, x=x, y=y, z=z)  # validate the input
        object_id = "O" + str(len(self.objects) + 1)  # determine the new object's ID
        self.objects[object_id] = Node(object_class=object_class, x=x, y=y, z=z)  # create the new node
        if object_class not in self.__objects_by_class__:  # check if the objects by class has this class
            self.__objects_by_class__[object_class] = []  # init the class
        self.__objects_by_class__[object_class].append(object_id)  # add the object ID to the class list
        return
    
    # remove an object from the scene graph
    def remove_object(self, object_id:str):
        if object_id not in self.objects:
            raise KeyError("Given object ID (" + str(object_id) + ") is not in the node table.")
        object_class = self.objects[object_id].object_class
        del self.objects[object_id]  # remove the object
        self.__objects_by_class__[object_class].remove(object_id)  # remove the object from the class list
        return

    # parses a set of visible objects to update the scene
    # used when the object ID is not known, this function figures out which object we are changing
    # seen_objects: list[dict] : [{class : str, x : int, y : int, z : int, last seen : int}]
    def update(self, seen_objects:list):
        # validate the input
        self.__validate_seen_object_list__(seen_objects)
        # check through the cases in the resolver, or give up and place the object in the unresolved bin
        unresolved_seen_objects = []  # seen object dicts that are unresolved
        resolved_known_objects = []  # known object IDs that are resolved
        # check the naive match: simple "same class, similar location" check
        for seen_object in seen_objects:
            found, known_object_id = self.__find_naive_match__(seen_object["class"], seen_object["x"], seen_object["y"], seen_object["z"])  # get a match if one exists
            if found:  # found a match, so update the object and mark it as resolved
                self.objects[known_object_id].update(x=seen_object["x"], y=seen_object["y"], z=seen_object["z"], last_seen=seen_object["last seen"] if "last seen" in seen_object else None)  # update the object
                resolved_known_objects.append(known_object_id)  # add the object to the resolved objects
            else:  # could not find a match, so mark the seen object as unresolved
                unresolved_seen_objects.append(seen_object)

        # for the remaining unresolved, get the closest object of their class
        closest_objects_by_class = {}  # dictionary of class : known object ID : [[seen object ID, distance]], used to resolve objects and distances
        closest_objects_recorded = []  # list of known object IDs that have been recorded so far, so finding closest matches don't result in duplicates'
        for unresolved_seen_object in unresolved_seen_objects:
            # get the closest known object of that class
            closest_object_id, distance = self.__find_closest_match__(unresolved_seen_object["class"], unresolved_seen_object["x"], unresolved_seen_object["y"], unresolved_seen_object["z"], ignore=resolved_known_objects + closest_objects_recorded)
            if closest_object_id is not None:  # if found a closest object, add it to the close object list
                if unresolved_seen_object["class"] not in closest_objects_by_class:  # ensure the class is in the distance dict
                    closest_objects_by_class[unresolved_seen_object["class"]] = {}
                if closest_object_id not in closest_objects_by_class[unresolved_seen_object["class"]]:  # ensure the known object ID is in the distance dict
                    closest_objects_by_class[unresolved_seen_object["class"]][closest_object_id] = []
                closest_objects_by_class[unresolved_seen_object["class"]][closest_object_id].append([unresolved_seen_object, distance])  # add to the distance dict
                closest_objects_recorded.append(closest_object_id)  # add to the record of closest objects

        # for each known object, pick the seen object that claimed it that is closest of the known object
        for object_class in closest_objects_by_class:  # resolve each known object
            for known_object_id in closest_objects_by_class[object_class]:
                [closest_seen_object, _] = min(closest_objects_by_class[object_class][known_object_id], key = lambda x : x[1])  # get the closest object
                found, idx = self.__check_if_object_in_list__(closest_seen_object, unresolved_seen_objects)
                if found:
                    self.objects[known_object_id].update(x=closest_seen_object["x"], y=closest_seen_object["y"], z=closest_seen_object["z"], last_seen=closest_seen_object["last seen"] if "last seen" in closest_seen_object else None)  # update the known object
                    resolved_known_objects.append(known_object_id)  # set the known object as resolved
                    del unresolved_seen_objects[idx]  # remove the seen object from unresolved

        # raise an error if there are still unresolved objects
        if len(unresolved_seen_objects) > 0:
            raise ValueError("DSG solver could not resolve all seen objects: " + str([x["class"] + " " + str(round(x["x"], 2)) + " " + str(round(x["y"], 2)) + " " + str(round(x["z"], 2)) for x in unresolved_seen_objects]) + " // all objects: " + str(resolved_known_objects) + " // seen objects: " + str([x["class"] + " " + str(round(x["x"], 2)) + " " + str(round(x["y"], 2)) + " " + str(round(x["z"], 2)) for x in seen_objects])) # + str([x["class"] + " " + str(round(x["x"], 2)) + " " + str(round(x["y"], 2)) + " " + str(round(x["z"], 2)) for x in resolved_known_objects]))

        return

    # get the number of objects in the scene graph
    def count(self):
        return len(self.objects)
    
    # get the objetst by class
    def get_objects_by_class(self) -> dict:
        objects_by_class = {}
        for object_id in self.objects:
            object_class = self.objects[object_id].object_class
            if object_class not in objects_by_class:
                objects_by_class[object_class] = []
            objects_by_class[object_class].append(self.objects[object_id])
        return objects_by_class

    # update a known object's properties, used when the object ID is known
    def __update_known_object__(self, object_id:int, object_class:Optional[str]=None, x:Optional[float]=None, y:Optional[float]=None, z:Optional[float]=None, last_seen:Optional[float]=None):
        if object_id not in self.objects:  # check if the node exists
            raise KeyError("Given object ID (" + str(object_id) + ") is not in the node table.")
        self.objects[object_id].update(object_class, x, y, z, last_seen)  # update the node
        return

    # check if a list of objects contains a given object, used so numpy arrays aren't compared
    def __check_if_object_in_list__(self, o, l):
        for i in range(len(l)):
            if l[i]["class"] == o["class"] and l[i]["x"] == o["x"] and l[i]["y"] == o["y"] and l[i]["z"] == o["z"]:
                return True, i
        return False, -1

    # finds an object of the same class and similar location
    def __find_naive_match__(self, object_class:str, x:float, y:float, z:float) -> tuple[bool, str]:
        # check if the object class is known
        if object_class not in self.__objects_by_class__:
            raise KeyError("The given object class '" + str(object_class) + "' is not in the known objects list.")

        # naive case: same class, same location
        for object_id in self.__objects_by_class__[object_class]:  # for each known object in the seen object's class'
            if self.objects[object_id].distance(x, y, z) <= self.same_location_threshold:  # if the two objects share a location
                return True, object_id  # found a match

        return False, ""  # no naive match

    # finds the closest object of the class
    def __find_closest_match__(self, object_class:str, x:float, y:float, z:float, ignore:list=[]) -> tuple[str | None, float]:
        # check if the object class is known
        if object_class not in self.__objects_by_class__:
            raise KeyError("The given object class '" + str(object_class) + "' is not in the known objects list.")

        # get the closest known object of the correct class
        closest_distance = float("infinity")
        closest_object_id = None
        for object_id in self.__objects_by_class__[object_class]:  # for each known object in the seen object's class'
            if object_id not in ignore:  # only track objects not in the ignore list (resolved elsewhere)
                dist = self.objects[object_id].distance(x, y, z)  # get the distance
                if dist <= closest_distance:  # if the two objects share a location
                    closest_distance = dist
                    closest_object_id = object_id

        return closest_object_id, closest_distance

    # check arguments of a function that can take either an object dict or individual properties
    def __validate_object_properties__(self, object_dict:Optional[dict]=None, object_class:Optional[str]=None, x:Optional[float]=None, y:Optional[float]=None, z:Optional[float]=None) -> tuple[str | None, float | None, float | None, float | None]:
        # if the object dictionary was not passed in, ensure all the object properties were passed in
        if object_dict is None:
            if object_class is None or not isinstance(x, str):
                raise ValueError("Validating object properties: an object dictionary was not passed in and the object_class param is not a string, fix by either passing a dictionary to object_dict or a string to object_class.")
            if x is None or not isinstance(x, float):
                raise ValueError("Validating object properties: an object dictionary was not passed in and the x param is not a number, fix by either passing a dictionary to object_dict or a number to x.")
            if y is None or not isinstance(x, float):
                raise ValueError("Validating object properties: an object dictionary was not passed in and the y param is not a number, fix by either passing a dictionary to object_dict or a number to y.")
            if z is None or not isinstance(x, float):
                raise ValueError("Validating object properties: an object dictionary was not passed in and the z param is not a number, fix by either passing a dictionary to object_dict or a number to z.")
        # if the object dictionary was passed in, ensure no object properties were passed in
        else:
            # ensure no object properties were passed in
            if object_class is not None:
                raise ValueError("Validating object properties: an object dictionary was passed in and an object_class param was passed in, fix by only passing in one of those parameters.")
            if x is not None:
                raise ValueError("Validating object properties: an object dictionary was passed in and an x param was passed in, fix by only passing in one of those parameters.")
            if y is not None:
                raise ValueError("Validating object properties: an object dictionary was passed in and a y param was passed in, fix by only passing in one of those parameters.")
            if z is not None:
                raise ValueError("Validating object properties: an object dictionary was passed in and a z param was passed in, fix by only passing in one of those parameters.")
            # ensure the object dictionary has the needed properties
            if "x" not in object_dict:
                raise ValueError("Validating object properties: an object dictionary was passed in but it does not have an 'x' field: " + str(object_dict["class"]))
            if "y" not in object_dict:
                raise ValueError("Validating object properties: an object dictionary was passed in but it does not have a 'y' field: " + str(object_dict["class"]))
            if "z" not in object_dict:
                raise ValueError("Validating object properties: an object dictionary was passed in but it does not have a 'z' field: " + str(object_dict["class"]))
            if "class" not in object_dict:
                raise ValueError("Validating object properties: an object dictionary was passed in but it does not have a 'class' field: " + str(object_dict["class"]))
            # set the variables to the object properties
            object_class = object_dict["class"]
            x = object_dict["x"]
            y = object_dict["y"]
            z = object_dict["z"]
        # return the properties
        return object_class, x, y, z

    # check structure of the seen object list, fails if at least one object has an invalid form
    def __validate_seen_object_list__(self, seen_objects:list) -> None:
        for seen_object in seen_objects:
            # error if seen object is not a dictionary
            if not isinstance(seen_object, dict):
                raise ValueError("Validating seen objects: object is not a dict: " + str(seen_object))
            self.__validate_object_properties__(object_dict=seen_object)  # validate the input, will error if the dictionary is mal-formed
        return

    # resets the DSG
    def __reset__(self) -> None:
        self.objects.clear()
        self.lost_object_ids.clear()
        self.__objects_by_class__.clear()
        return

    # returns a string representation of the DSG
    def __str__(self) -> str:
        return str({k : self.objects[k].as_dict() for k in self.objects})

    # returns the object with the given key
    def __getitem__(self, key) -> Node:
        if key not in self.objects:
            raise KeyError("The given object ID '" + str(key) + "' is not in the known objects dictionary.")
        return self.objects[key]
