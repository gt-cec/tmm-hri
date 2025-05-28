from virtualhome.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication, UnityCommunicationException, UnityEngineException
from virtualhome.virtualhome.simulation.unity_simulator import utils_viz
# from virtualhome.virtualhome.demo.utils_demo import *
import glob
from PIL import Image
import random, threading, time, subprocess, datetime, os, pathlib
import platform, math
import pickle, utils

comm = None
random.seed(datetime.datetime.now().timestamp())

ignore_objects = ["lime", "waterglass", "slippers", "cellphone"]  # limes/cellphones have trouble with interaction positions, waterglass have problems with IDs sticking to the graph, slippers are on the ground
os_name = platform.system()

room_points = {
    "bathroom": ((-8.75, 1.11), (-8.48, -.4)),
    "bedroom": ((-7.11, -4.99), (-7.7, -5.96)),
    "kitchen": ((-4.87, -6.91), (-0.64, -5.16)),
    "livingroom": ((5.85, -6.9), (5.90, -6.9))
}

def walkthrough_household(output_folder:str="Episodes", file_name_prefix="Current"):
    num_traversed_rooms = 0
    rooms, _, objects_in_rooms, _ = __get_objects_in_rooms__()
    room_ids = list(rooms.keys())
    #for _ in range(10):
    for i in range(len(rooms)):
        target_objects, success, sim_failure = __sim_action__("walk", room_points[rooms[room_ids[i]][1]], list(objects_in_rooms[room_ids[i]]), output_folder=output_folder, file_name_prefix=file_name_prefix)
        if sim_failure:
            return False, num_traversed_rooms
        if success:
            num_traversed_rooms += 1
        else:  # recovery: failed to reach surface, choose another surface
            pass
    return True, num_traversed_rooms

# remove duplicate items from the graph
def __remove_duplicate_items_from_graph__(g:dict):
    # removes duplicate classes (or virtualhome sometimes won't get the agent )
    new_graph = {"nodes": [], "edges": []}
    used_classes = set()
    used_ids = set()
    nodes_inside_of_others = set()
    rooms = set()
    for n in g["nodes"]:  # get the rooms so we ignore their INSIDE relations
        if n["category"] == "Rooms":
            rooms.add(n["id"])

    for e in g["edges"]:  # record nodes that are inside of things
        if e["relation_type"] == "INSIDE" and e["to_id"] not in rooms:
            nodes_inside_of_others.add(e["from_id"])
            
    for n in g["nodes"]:  # keep nodes that aren't grabbable or one grabbable node per class
        if ("GRABBABLE" not in n["properties"] or n["class_name"] not in used_classes) and n["id"] not in nodes_inside_of_others:
            new_graph["nodes"].append(n)
            # used_classes.add(n["class_name"])
            used_ids.add(n["id"])

    for e in g["edges"]:  # keep edges between two valid nodes
        if e["from_id"] in used_ids and e["to_id"] in used_ids:
            new_graph["edges"].append(e)
    return new_graph

def __randomize_object_locations__(g:dict) -> dict:
    """
    Randomizes the object locations in the graph.

    Args:
        g (dict): The graph to randomize the object locations in.

    Returns:
        dict: The graph with the object locations randomized.
    """
    positions_ids = []  # ids of the objects
    positions_positions = []  # positions of the bounding box center
    positions_names = []  # class name of the object
    
    # get the positions/ids/classes of all grabbable objects
    for n in g["nodes"]:
        if not utils.__node_is_grabbable__(n, ignore_objects=ignore_objects):
            continue
        positions_ids.append(n["id"])
        positions_positions.append(n["bounding_box"]["center"])
        positions_names.append(n["class_name"])

    # get the objects in each room
    rooms, _, objects_in_rooms, g = __get_objects_in_rooms__()
    positions_ids_old = [x for x in positions_ids]

    # shuffle the positions IDs
    random.shuffle(positions_ids)

    positions_mapping = {}
    for i in range(len(positions_ids)):
        # use the shuffled object IDs so we can go through and assign each object the object ID and position of a different object
        # positions mapping[original object ID] = (new object ID, position of new object ID, class of new object ID)
        positions_mapping[positions_ids_old[i]] = (positions_ids[i], positions_positions[i], positions_names[i])
    
    # replace the positions
    for id_of_node_to_change in positions_mapping:
        # the object ID that the current object will take the position of
        using_position_of_id = positions_mapping[id_of_node_to_change][0]
        # find the room of the old object, remove the connection to the old object, create the connection to the new object
        for room in objects_in_rooms:  # for each room
            for item in objects_in_rooms[room]:  # for each item in the room
                # if the item ID matches the object ID that the current object will take the position of
                # if item[0] == positions_mapping[using_position_of_id][0]:    # removed because items werent shuffling
                if item[0] == using_position_of_id:
                    # remove previous INSIDE
                    g["edges"].remove({"from_id": item[0], "to_id": room, "relation_type": "INSIDE"})
                    # add new INSIDE
                    g["edges"].append({"from_id": id_of_node_to_change, "to_id": room, "relation_type": "INSIDE"})

        # change the room
        for i in range(len(g["nodes"])):
            if g["nodes"][i]["id"] == id_of_node_to_change:
                g["nodes"][i]["bounding_box"]["center"] = positions_mapping[using_position_of_id][1]
                g["nodes"][i]["obj_transform"]["position"] = positions_mapping[using_position_of_id][1]
                break
    return g

# kill the simulator to get as fresh run, a bash script on the server should have it restart automatically
def __reset_sim__(seed=42):
    global comm
    comm = UnityCommunication(no_graphics=False, port=8080)  # set up communiciation with the simulator, I don't think no_graphics actually does anything
    while True:  # keep trying to reconnect
        try:
            comm.reset()
            break
        except Exception as e:
            print("Waiting for simulator to accept a connection. Exception details:", str(e))
            time.sleep(3)
    
    random.seed(seed)
    # remove items that are too far or duplicate IDs
    g = comm.environment_graph()[1]
    g = __remove_duplicate_items_from_graph__(g)
    comm.expand_scene(g)
    time.sleep(3)
    # randomize the object locations
    g = comm.environment_graph()[1]
    preshuffle_nodes = [x for x in g["nodes"]]
    g1 = {k:g[k] for k in g}
    g = __randomize_object_locations__(g)
    with open("g1.txt", "w") as f:
        f.write(str(g1))
    with open("g2.txt", "w") as f:
        f.write(str(g))
    
    
    comm.expand_scene(g)
    time.sleep(3)
    comm.add_character('Chars/Male2', initial_room='kitchen')
    time.sleep(3)
    comm.add_character('Chars/Male2', initial_room='kitchen')

    instance_colormap = {}
    obj_colors = comm.instance_colors()[1]
    _, g = comm.environment_graph()
    for n in g["nodes"]:
        rgb_color = str(obj_colors[str(n["id"])][0] * 255) + "," + str(obj_colors[str(n["id"])][1] * 255) + "," + str(obj_colors[str(n["id"])][2] * 255)
        instance_colormap[rgb_color] = (n["id"], n["class_name"], n["bounding_box"]["center"])
    return instance_colormap, obj_colors, g, preshuffle_nodes

def __object_of_min_dist__(loc, objects):
    min_dist = float("infinity")
    min_i = None
    for i in range(len(objects)):
        dist = math.sqrt((loc[0] - objects[i][2][0]) ** 2 + (loc[1] - objects[i][2][2]) ** 2)
        print("  compare object", objects[i], "loc", loc, "dist", dist, "min?", dist < min_dist)
        if dist < min_dist:
            min_i = i
            min_dist = dist
    print("object of min dist", objects[min_i], "dist", min_dist, "target loc", loc)
    return objects[min_i]

# sends an action to all agents
def __sim_action__(action:str, room_points:list, objects_in_room:list, char_ids:list=[0,1], object_ids:list=None, output_folder:str="Output/", file_name_prefix:str="script"):
    sim_failure = False  # flag for the sim failing, requires restart
    object_ids = [__object_of_min_dist__(room_points[i], objects_in_room) for i in range(len(room_points))]

    for agent_id in reversed(char_ids):
        script = f"<char{agent_id}> [{action}] <{object_ids[agent_id][1]}> ({object_ids[agent_id][0]})"
        print("Running script:", script, "saving to", output_folder, file_name_prefix)
        success, sim_failure = __sim_script__(script, output_folder=output_folder, file_name_prefix=file_name_prefix)
        if not success:
            print("FAILURE", sim_failure)
            break
        print("Success!", success, sim_failure)
        time.sleep(3)
    
    script = f"<char0> [lookat] <{object_ids[1][1]}> ({object_ids[1][0]})"
    print("Running script:", script)
    success, sim_failure = __sim_script__(script, output_folder=output_folder, file_name_prefix=file_name_prefix)
    return object_ids, success, sim_failure

# run an action on the simulator
def __sim_script__(script:str, output_folder:str="Output/", file_name_prefix:str="script"):
    sim_failure = False  # flag for the sim failing, requires restart
    while True:  # keep trying in case there are connection errors
        try:  # try running the script
            success, message = comm.render_script([script], 
                                                  image_synthesis=["normal", "seg_inst", "seg_class", "depth"], 
                                                  camera_mode=["FIRST_PERSON"], 
                                                  image_width=512, 
                                                  image_height=512, 
                                                  save_pose_data=True, 
                                                  recording=True, 
                                                  find_solution=False,
                                                  processing_time_limit=10000, 
                                                  frame_rate=10, 
                                                  output_folder=output_folder, 
                                                  file_name_prefix=file_name_prefix
                                                )
            if success:  # if a script execution failed, the success flag will still be True, so mark as failed
                success = False if len([True for x in message if message[x]["message"] != "Success"]) > 0 else success
            print(f"Script Complete: {success}, {message}")
            break
        except UnityCommunicationException as e:
            success = False
            print(f"Script Fail: {str(e)}")
            break
        except UnityEngineException as e:  # the engine or communication can fail, so restart everything
            sim_failure = True
            print(f"Engine Fail: {str(e)}")
        except Exception as e:
            success = False
            sim_failure = True
            print(f"Unknown reason why script Fail: {str(e)}")
            break
    return success, sim_failure

# pull the objects and surfaces from the graph in each room
def __get_objects_in_rooms__(g={}):
    if g == {}:
        _, g = comm.environment_graph() # get the environment graph
    rooms = {x["id"] : (x["id"], x["class_name"], x["bounding_box"]["center"]) for x in g["nodes"] if x["category"] == "Rooms"}
    objects = {x["id"] : (x["id"], x["class_name"], x["bounding_box"]["center"]) for x in g["nodes"] if utils.__node_is_grabbable__(x)}
    edges = {x["from_id"] : x for x in g["edges"] if x["relation_type"] == "INSIDE" and x["from_id"] in objects and x["to_id"] in rooms}
    objects_in_rooms = {room_id : [objects[edge["from_id"]] for edge in edges.values() if edge["to_id"] == room_id] for room_id in rooms}
    return rooms, objects, objects_in_rooms, g

# run the agents
if __name__ == "__main__":
    # get the first argument as the seed
    seed = 42
    if len(os.sys.argv) > 1:
        try:
            seed = int(os.sys.argv[1])
        except ValueError:
            print("Invalid seed argument, using default seed 42.")
    output_folder = "../episodes"  # relative to the executable, in our case up one directory
    episode_count = 1
    num_agents = 2
    for i in range(episode_count):  # run a fixed number of episodes so the dataset doesn't use all storage (1-2GB per run)
        episode_name = f"episode_{seed}"
        print("Starting", episode_name, output_folder + "/" + episode_name)
        print("Resetting sim...")
        # make episode directory if it doesn't exist
        pathlib.Path(f"episodes/{episode_name}").mkdir(parents=True, exist_ok=True)
        instance_colormap, object_colors, g, preshuffle_nodes = __reset_sim__(seed=seed)  # reload the simulator
        with open(f"episodes/{episode_name}/color_info.pkl", "wb") as f:
            pickle.dump((instance_colormap, object_colors, g), f)
        with open(f"episodes/{episode_name}/preshuffle_nodes.pkl", "wb") as f:
            pickle.dump(preshuffle_nodes, f)
        print("Sim reset! Walking through household now.")
        g = comm.environment_graph()[1]
        with open(f"episodes/{episode_name}/starting_graph.pkl", "wb") as f:
            pickle.dump(g, f)
        print("Starting graph saved.")
        res, num_traversed_rooms = walkthrough_household(output_folder=output_folder, file_name_prefix=episode_name)  # run the pick/place sim
        with open(f"episodes/{episode_name}/episode_info.txt", "w") as f:  # add an episode info file
            f.write(f"{episode_name}\n{res}\n\n{instance_colormap}")
        print("Completed agent run", episode_name, ": ended gracefully?", res)

print("Done!")
