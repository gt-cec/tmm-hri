from virtualhome.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication, UnityCommunicationException, UnityEngineException
from virtualhome.virtualhome.simulation.unity_simulator import utils_viz
from virtualhome.virtualhome.demo.utils_demo import *
import glob
from PIL import Image
import random, threading, time, subprocess, datetime, os
import platform
import utils

comm = None
random.seed(datetime.datetime.now().timestamp())

ignore_objects = ["lime", "waterglass", "slippers"]  # limes have trouble with interaction positions, waterglass have problems with IDs sticking to the graph, slippers are on the ground
ignore_surfaces = ["bookshelf", "bench"]  # bookshelves have a lot of occlusion, bench fails for a lot of placements

os_name = platform.system()

def move_agents(count, num_agents:int=2, output_folder:str="Episodes", file_name_prefix="Current"):
    rooms = __get_rooms__()  # get the rooms
    print(rooms)
    num_traversed_rooms = 0
    room_ids = list(rooms.keys())
    # shuffle object locations
    objects, surfaces, g = __get_objects_and_surfaces__()
    original_g = g.copy()
    new_g = {"nodes": [], "edges": []}
    shuffled_object_ids = list(objects.keys())
    random.shuffle(shuffled_object_ids)
    for i in range(len(shuffled_object_ids)):
        original_id = list(objects.keys())[i]
        new_id = shuffled_object_ids[i]
        # replace the edges
        for i in range(len(original_g["edges"])):
            if original_g["edges"][i]["from_id"] == original_id:
                new_g["edges"].append({"from_id": new_id, "to_id": original_g["edges"][i]["to_id"], "relation_type": original_g["edges"][i]["relation_type"]})
            if original_g["edges"][i]["to_id"] == original_id:
                new_g["edges"].append({"to_id": new_id, "from_id": original_g["edges"][i]["from_id"], "relation_type": original_g["edges"][i]["relation_type"]})
            
        # replace the location
        for i in range(len(original_g["nodes"])):
            if original_g["nodes"][i]["id"] == original_id:
                index_in_original = i
        new_node = original_g["nodes"][index_in_original].copy()
        new_node["id"] = new_id
        new_node["obj_transform"]["position"] = original_g["nodes"][index_in_original]["obj_transform"]["position"]
        g["nodes"].append(new_node)

    # update the graph
    print("Have new g!", g)
    comm.expand_scene(g)
    print("Shuffled graph!")

    rooms, objects, objects_in_rooms, g = __get_objects_in_rooms__()
    for i in range(len(rooms)):
        target_objects, success, sim_failure = __sim_action__("walk", object_ids=[], sample_source=list(objects_in_rooms[room_ids[i]]), output_folder=output_folder, file_name_prefix=file_name_prefix)
        print("Moved to next room", rooms[room_ids[i]], "Target objects", target_objects)
        if sim_failure:
            return False, num_traversed_rooms
        if success:
            num_traversed_rooms += 1
        else:  # recovery: failed to reach surface, choose another surface
            pass
    return True, num_traversed_rooms

# remove duplicate items from the graph
def __remove_duplicate_items_from_graph__(g:dict):
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
    comm.expand_scene(new_graph)
    return

# kill the simulator to get as fresh run, a bash script on the server should have it restart automatically
def __reset_sim__():
    print("Sending kill command to simulator")
    sim_filename = "macos_exec.v2.3.0.app" if os_name == "Darwin" else "linux_exec.v2.3.0.x86_64"
    subprocess.run(["pkill", "-f", sim_filename])
    print("  Sent, reconnecting - expect a few seconds of waiting to accept a connection while the simulator restarts")
    global comm
    comm = UnityCommunication(no_graphics=False)  # set up communiciation with the simulator, I don't think no_graphics actually does anything
    while True:  # keep trying to reconnect
        try:
            comm.reset()
            break
        except Exception as e:
            print("Waiting for simulator to accept a connection. Exception details:", str(e))
            time.sleep(3)
    
    __remove_duplicate_items_from_graph__(comm.environment_graph()[1])
    time.sleep(5)
    comm.add_character('Chars/Male2', initial_room='kitchen')  # add two agents this time
    comm.add_character('Chars/Male2', initial_room='kitchen')

    instance_colormap = {}
    obj_colors = comm.instance_colors()[1]
    _, g = comm.environment_graph()
    for n in g["nodes"]:
        rgb_color = str(obj_colors[str(n["id"])][0] * 256) + "," + str(obj_colors[str(n["id"])][1] * 256) + "," + str(obj_colors[str(n["id"])][2] * 256)
        instance_colormap[rgb_color] = (n["id"], n["class_name"], n["bounding_box"]["center"])
    return instance_colormap

# sends an action to all agents
def __sim_action__(action:str, char_ids:list=[0,1], object_ids:list=None, surface_ids:list=None, sample_source:str=None, output_folder:str="Output/", file_name_prefix:str="script"):
    sim_failure = False  # flag for the sim failing, requires restart
    if sample_source is not None and object_ids == []:  # if sampling the objects
        object_ids = __sample_objects__(sample_source, num=len(char_ids))
    elif sample_source is not None and surface_ids == []:  # if sampling the surfaces
        surface_ids = __sample_objects__(sample_source, num=len(char_ids))
    script = " | ".join([f"<char{agent_id}> [{action}]" + (f" <{object_ids[agent_id][1]}> ({object_ids[agent_id][0]})" if object_ids is not None and object_ids[agent_id] is not None else "") + (f" <{surface_ids[agent_id][1]}> ({surface_ids[agent_id][0]})" if surface_ids is not None and surface_ids[agent_id] is not None  else "") for agent_id in char_ids[0:]])
    print("Running script:", script)
    success, sim_failure = __sim_script__(script, output_folder=output_folder, file_name_prefix=file_name_prefix)
    if surface_ids is None and action not in ["grab", "put"]:
        return object_ids, success, sim_failure
    elif object_ids is None and action not in ["grab", "put"]:
        return surface_ids, success, sim_failure
    return success, sim_failure

# run an action on the simulator
def __sim_script__(script:str, output_folder:str="Output/", file_name_prefix:str="script"):
    sim_failure = False  # flag for the sim failing, requires restart
    while True:  # keep trying in case there are connection errors
        try:  # try running the script
            success, message = comm.render_script([script], image_synthesis=["normal", "seg_inst", "seg_class", "depth"], camera_mode=["FIRST_PERSON"], image_width=512, image_height=512, save_pose_data=True, recording=True, find_solution=False, processing_time_limit=10000, frame_rate=10, output_folder=output_folder, file_name_prefix=file_name_prefix)
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

# pull the rooms from the graph
def __get_rooms__() -> dict:
    _, g = comm.environment_graph()  # get the environment graph
    rooms = {x["id"] : (x["id"], x["class_name"], x["obj_transform"]["position"]) for x in g["nodes"] if x["category"] == "Rooms"}
    return rooms

# pull the objects and surfaces from the graph
def __get_objects_and_surfaces__() -> tuple[dict, dict, dict]:
    _, g = comm.environment_graph() # get the environment graph
    objects = {x["id"] : (x["id"], x["class_name"], x["obj_transform"]["position"]) for x in g["nodes"] if x["category"] == "Props" and "GRABBABLE" in x["properties"] and x["class_name"] not in ignore_objects}
    surfaces = {x["id"] : (x["id"], x["class_name"], x["obj_transform"]["position"], x["category"], x["properties"]) for x in g["nodes"] if x["category"] == "Furniture" and "SURFACES" in x["properties"] and "GRABBABLE" not in x["properties"] and "CAN_OPEN" not in x["properties"] and x["class_name"] not in ignore_surfaces}
    return objects, surfaces, g

# pull the objects and surfaces from the graph in each room
def __get_objects_in_rooms__() -> tuple[dict, dict, dict]:
    _, g = comm.environment_graph() # get the environment graph
    rooms = {x["id"] : (x["id"], x["class_name"], x["obj_transform"]["position"]) for x in g["nodes"] if x["category"] == "Rooms"}
    objects = {x["id"] : (x["id"], x["class_name"], x["obj_transform"]["position"]) for x in g["nodes"] if x["category"] == "Props" and "GRABBABLE" in x["properties"] and x["class_name"] not in ignore_objects}
    edges = {x["from_id"] : x for x in g["edges"] if x["relation_type"] == "INSIDE" and x["from_id"] in objects and x["to_id"] in rooms}
    objects_in_rooms = {room_id : [objects[edge["from_id"]] for edge in edges.values() if edge["to_id"] == room_id] for room_id in rooms}
    return rooms, objects, objects_in_rooms, g

# sample two items, choose items that are not close together so the agents don't get stuck 
def __sample_objects__(sample_source:list, num:int=2, max_dist:float=3):
    # bit of a hack, just resample until we get objects that aren't very close to each other
    # to optimize: start with a sample, and then resample objects that are close to each other
    escape_limit = 1000  # after 1000 attempts we know something is dearly wrong
    while escape_limit > 0:  # will continue until two good objects are found
        res = random.sample(sample_source, num)  # generate original sample, without replacement
        failed = False  # flag for whether the checks failed
        for obj_1_idx in range(num):  # check each object pair for distance violations
            obj_1_res = res[obj_1_idx]
            for obj_2_idx in range(obj_1_idx+1, num):
                obj_2_res = res[obj_2_idx]
                if (obj_1_res[2][0] - obj_2_res[2][0]) ** 2 + (obj_1_res[2][2] - obj_2_res[2][2]) ** 2 < max_dist ** 2:
                    failed = True
                    break
            if failed:
                break
        if not failed:
            break
        escape_limit -= 1  # decrement the attempts remaining
    return res

# run the agents
if __name__ == "__main__":
    output_folder = "episodes"
    episode_count = 1
    num_agents = 2
    for i in range(episode_count):  # run a fixed number of episodes so the dataset doesn't use all storage (1-2GB per run)
        episode_name = f"episode_{datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")}_agents_{num_agents}_run_{i}"
        if os.path.isdir(output_folder + "/" + episode_name):  # if the episode already exists, overwrite it (e.g., did not complete enough cycles)
            subprocess.run(["rm", "-rf", output_folder + "/" + episode_name])
            print("Removed previous run because it did not complete enough cycles:", num_complete)
        print("Starting", episode_name)
        instance_colormap = __reset_sim__()  # reload the simulator
        res, num_complete = move_agents(500, num_agents=num_agents, output_folder=output_folder, file_name_prefix=episode_name)  # run the pick/place sim
        with open(output_folder + "/" + episode_name + "/episode_info.txt", "w") as f:  # add an episode info file
            f.write(f"{episode_name}\n{num_complete}\n{res}\n{instance_colormap}")
        print("Completed agent run", episode_name, ": ended gracefully?", res, "Completed", num_complete)

print("Done!")
