from virtualhome.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication, UnityCommunicationException, UnityEngineException
from virtualhome.virtualhome.simulation.unity_simulator import utils_viz
import glob
import random, threading, time, subprocess, datetime, os
import platform
import types

comm = None
random.seed(datetime.datetime.now().timestamp())

ignore_objects = ["lime", "waterglass", "slippers"]  # limes have trouble with interaction positions, waterglass have problems with IDs sticking to the graph
ignore_surfaces = ["bookshelf", "bench"]  # bookshelves have a lot of occlusion, bench fails for a lot of placements

os_name = platform.system()

def move_agents(count, num_agents:int=2, output_folder:str="Episodes", file_name_prefix="Current"):
    state = "walk to object"
    completed_relocations = 0
    success = True
    while completed_relocations < count:  # run a state machine to relocate objects
        print("STARTING NEXT RELOCATION", completed_relocations)
        if success:  # if last element was successful, reload the graph
            objects, surfaces, g = __get_objects_and_surfaces__()  # pull the latest objects, surfaces, and the environment graph
        if state == "walk to object":  # go to an object
            target_objects, success, sim_failure = __sim_action__("walk", object_ids=[], sample_source=list(objects.values()), output_folder=output_folder, file_name_prefix=file_name_prefix)
            if sim_failure:
                return False, completed_relocations
            if success:
                state = "grab"
            else:  # recovery: failed to walk to object, choose another object
                pass
        elif state == "grab":  # pick it up
            success, sim_failure = __sim_action__("grab", object_ids=target_objects, output_folder=output_folder, file_name_prefix=file_name_prefix)
            if sim_failure:
                return False, completed_relocations
            if success:
                state = "walk to surface"
            else:  # recovery: failed to grab object, choose another object
                print("GRAB failed, recovering by putting and then going to another object")
                state = "walk to object"
        elif state == "walk to surface":  # go to an surface
            target_surfaces, success, sim_failure = __sim_action__("walk", surface_ids=[], sample_source=list(surfaces.values()), output_folder=output_folder, file_name_prefix=file_name_prefix)
            if sim_failure:
                return False, completed_relocations
            if success:
                state = "put"
            else:  # recovery: failed to reach surface, choose another surface
                pass
        elif state == "put":  # place it down
            # get the objects are currently held
            held_objects = {}
            for obj in [x for x in g["edges"] if x["relation_type"] == "HOLDS_LH" or x["relation_type"] == "HOLDS_RH"]:
                holder = obj["from_id"] - 1
                if holder not in held_objects:
                    held_objects[holder] = []
                held_objects[holder].append(obj["to_id"])
            objects_to_place = [objects[held_objects[char_id][0]] if char_id in held_objects and len(held_objects[char_id]) > 0 and held_objects[char_id][0] in objects else None for char_id in range(num_agents)]  # get the object dict corresponding to the first held object of each character
            char_ids = [char_id for char_id in range(num_agents) if objects_to_place[char_id] != None]
            # print("OBJECTS", objects_to_place, held_objects)
            success, sim_failure = __sim_action__("put", object_ids=objects_to_place, surface_ids=target_surfaces, char_ids=char_ids, output_folder=output_folder, file_name_prefix=file_name_prefix)
            if sim_failure:
                return False, completed_relocations
            if success:
                completed_relocations += 1
                state = "walk to object"
            else:  # recovery: failed to place object, choose another surface
                print("PUT failed, recovering by going to another surface")
                state = "walk to surface"
        else:
            raise ValueError("INVALID STATE!")
        # place down remaining
    return True, completed_relocations

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
    # print("Sending kill command to simulator")
    # sim_filename = "macos_exec.v2.3.0.app" if os_name == "Darwin" else "linux_exec.v2.3.0.x86_64"
    # subprocess.run(["pkill", "-f", sim_filename])
    # print("  Sent, reconnecting - expect a few seconds of waiting to accept a connection while the simulator restarts")
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
    if surface_ids is None and action not in ["grab", "put"]:
        return object_ids, success, sim_failure
    elif object_ids is None and action not in ["grab", "put"]:
        return surface_ids, success, sim_failure
    return success, sim_failure

# pull the objects and surfaces from the graph
def __get_objects_and_surfaces__():
    _, g = comm.environment_graph() # get the environment graph
    objects = {x["id"] : (x["id"], x["class_name"], x["bounding_box"]["center"]) for x in g["nodes"] if x["category"] == "Props" and "GRABBABLE" in x["properties"] and x["class_name"] not in ignore_objects}
    surfaces = {x["id"] : (x["id"], x["class_name"], x["bounding_box"]["center"], x["category"], x["properties"]) for x in g["nodes"] if x["category"] == "Furniture" and "SURFACES" in x["properties"] and "GRABBABLE" not in x["properties"] and "CAN_OPEN" not in x["properties"] and x["class_name"] not in ignore_surfaces}
    return objects, surfaces, g

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
                print("dist", obj_1_res, obj_2_res, (obj_1_res[2][0] - obj_2_res[2][0]), (obj_1_res[2][2] - obj_2_res[2][2]), (obj_1_res[2][0] - obj_2_res[2][0]) ** 2 + (obj_1_res[2][2] - obj_2_res[2][2]) ** 2, max_dist ** 2)
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
        episode_name = f"episode_{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')}_agents_{num_agents}_run_{i}"
        num_complete = 0  # number of pick/place cycles completed
        while num_complete < 5:  # repeat until the sim succeeds at 5+ cycles
            if os.path.isdir(output_folder + "/" + episode_name):  # if the episode already exists, overwrite it (e.g., did not complete enough cycles)
                subprocess.run(["rm", "-rf", output_folder + "/" + episode_name])
                print("Removed previous run because it did not complete enough cycles:", num_complete)
            print("Starting", episode_name)
            instance_colormap = __reset_sim__()  # reload the simulator
            res, num_complete = move_agents(500, num_agents=num_agents, output_folder=output_folder, file_name_prefix=episode_name)  # run the pick/place sim
            with open(output_folder + "/" + episode_name + "/episode_info.txt", "w") as f:  # add an episode info file
                f.write(f"{episode_name}\n{num_complete}\n{res}\n\n{instance_colormap}")
            print("Completed agent run", episode_name, ": ended gracefully?", res, "Completed", num_complete)
    
print("Done!")