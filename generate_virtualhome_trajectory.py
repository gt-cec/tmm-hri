from virtualhome.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication
from virtualhome.virtualhome.simulation.unity_simulator import utils_viz
from virtualhome.virtualhome.demo.utils_demo import *
import glob
from PIL import Image
import random, threading, time, subprocess

comm = None

def move_agents(count):
    state = "walk to object"
    completed_relocations = 0
    while completed_relocations < count:  # run a state machine to relocate objects
        objects, surfaces, g = __get_objects_and_surfaces__()  # pull the latest objects, surfaces, and the environment graph
        if state == "walk to object":  # go to an object
            target_objects, success, sim_failure = __sim_action__("walk", object_ids=[], sample_source=objects)
            if sim_failure:
                return False
            if success:
                state = "grab"
            else:  # recovery: failed to walk to object, choose another object
                pass
        elif state == "grab":  # pick it up
            success, sim_failure = __sim_action__("grab", object_ids=target_objects)
            if sim_failure:
                return False
            if success:
                state = "walk to surface"
            else:  # recovery: failed to grab object, choose another object
                print("GRAB failed, recovering by putting and then going to another object")
                state = "walk to object"
        elif state == "walk to surface":  # go to an surface
            target_surfaces, success, sim_failure = __sim_action__("walk", surface_ids=[], sample_source=surfaces)
            if sim_failure:
                return False
            if success:
                state = "put"
            else:  # recovery: failed to reach surface, choose another surface
                pass
        elif state == "put":  # place it down
            # get the objects are currently held
            held_objects = [x for x in g["edges"] if x["relation_type"] == "HOLDS_LH" or x["relation_type"] == "HOLDS_RH"]
            print("OBJECTS", held_objects)
            success, sim_failure = __sim_action__("put", object_ids=target_objects, surface_ids=target_surfaces)
            if sim_failure:
                return False
            if success:
                completed_relocations += 1
                state = "walk to object"
            else:  # recovery: failed to place object, choose another surface
                print("PUT failed, recovering by going to another surface")
                state = "walk to surface"
        else:
            raise ValueError("INVALID STATE!")
        # place down remaining
    return True

# kill the simulator to get as fresh run, a bash script on the server should have it restart automatically
def __reset_sim__():
    print("Sending kill command to simulator")
    subprocess.run(["pkill", "-f", 'linux_exec.v2.3.0.x86_64'])
    print("  Sent, reconnecting - expect a few seconds of waiting to accept a connection while the simulator restarts")
    global comm
    comm = UnityCommunication()  # set up communiciation with the simulator
    while True:  # keep trying to reconnect
        try:
            comm.reset()
            break
        except Exception as e:
            print("Waiting for simulator to accept a connection. Exception details:", str(e))
            time.sleep(3)
    comm.add_character('Chars/Male2', initial_room='kitchen')  # add two agents this time
    comm.add_character('Chars/Female4', initial_room='kitchen')
    return

# sends an action to all agents
def __sim_action__(action:str, num_agents:int=2, object_ids:list=None, surface_ids:list=None, sample_source:str=None):
    sim_failure = False  # flag for the sim failing, requires restart
    if sample_source is not None and object_ids == []:  # if sampling the objects
        object_ids = __sample_objects__(sample_source, num=num_agents)
    elif sample_source is not None and surface_ids == []:  # if sampling the surfaces
        surface_ids = __sample_objects__(sample_source, num=num_agents)
    script = " | ".join([f"<char{agent_id}> [{action}]" + (f" <{object_ids[agent_id][1]}> ({object_ids[agent_id][0]})" if object_ids is not None and object_ids[agent_id] is not None else "") + (f" <{surface_ids[agent_id][1]}> ({surface_ids[agent_id][0]})" if surface_ids is not None and surface_ids[agent_id] is not None  else "") for agent_id in range(num_agents)])
    print("Running script:", script)
    while True:  # keep trying in case there are connection errors
        try:  # try running the script
            success, message = comm.render_script([script], recording=True, find_solution=True, processing_time_limit=10000, frame_rate=10, camera_mode=["PERSON_FROM_BACK"])
            if success:  # if a script execution failed, the success flag will still be True, so mark as failed
                success = False if len([True for x in message if message[x]["message"] != "Success"]) > 0 else success
            print(f"Script Complete: {success}, {message}")
            break
        except Exception as e:  # the engine or communication can fail, so restart everything
            sim_failure = True
            print(f"Script Fail: {str(e)}")
    if surface_ids is None and action not in ["grab", "put"]:
        return object_ids, success, sim_failure
    elif object_ids is None and action not in ["grab", "put"]:
        return surface_ids, success, sim_failure
    return success, sim_failure

# pull the objects and surfaces from the graph
def __get_objects_and_surfaces__():
    _, g = comm.environment_graph() # get the environment graph
    objects = [(x["id"], x["class_name"], x["obj_transform"]["position"]) for x in g["nodes"] if x["category"] == "Props" and "GRABBABLE" in x["properties"]]
    surfaces = [(x["id"], x["class_name"], x["obj_transform"]["position"], x["category"], x["properties"]) for x in g["nodes"] if x["category"] == "Furniture" and "SURFACES" in x["properties"] and "GRABBABLE" not in x["properties"] and "CAN_OPEN" not in x["properties"]]
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
    __reset_sim__()
    res = move_agents(30)
    print("Completed agent run, ended gracefully?", res)


print("Done!")