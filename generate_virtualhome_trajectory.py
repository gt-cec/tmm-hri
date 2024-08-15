from virtualhome.virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication, UnityCommunicationException, UnityEngineException
from virtualhome.virtualhome.simulation.unity_simulator import utils_viz
from virtualhome.virtualhome.demo.utils_demo import *
import glob
from PIL import Image
import random, threading, time

comm = UnityCommunication()  # set up communiciation with the simulator

# reset the simulator
while True:
    try:
        comm.reset()
        break
    except UnityCommunicationException:
        print("Could not connect to simulator, is it running?")
        time.sleep(3)

s, g = comm.environment_graph() # get the environment graph

# add two agents this time
comm.add_character('Chars/Male2', initial_room='kitchen')
comm.add_character('Chars/Female4', initial_room='bedroom')

# print([x for x in g["edges"]])
objects = [(x["id"], x["class_name"], x["position"]) for x in g["nodes"] if x["category"] == "Props" and "GRABBABLE" in x["properties"]]
surfaces = [(x["id"], x["class_name"], x["category"], x["properties"]) for x in g["nodes"] if x["category"] == "Furniture" and "SURFACES" in x["properties"] and "GRABBABLE" not in x["properties"] and "CAN_OPEN" not in x["properties"]]
print(f"Relocating {len(objects)} objects")
print(objects)

def move_agents(count):
    for i in range(count):
        # go to an object
        target_objects = __sim_action__("walk", objects=[], sample_source=objects)
        # pick it up
        __sim_action__("grab", objects=target_objects)
        # go to an surface
        target_surfaces = __sim_action__("walk", surfaces=[], sample_source=surfaces)
        # edge closer
        __sim_action__("walktowards", surfaces=target_surfaces)
        # place it down
        __sim_action__("put", objects=target_objects, surfaces=target_surfaces)
        # place down remaining
        print(">>>", [x for x in g["edges"] if x["relation_type"] == "HOLDS_LH" or x["relation_type"] == "HOLDS_RH"])
    return

# sends an action to all agents
def __sim_action__(action:str, num_agents:int=2, objects:list=None, surfaces:list=None, sample_source:str=None):
    if sample_source is not None and objects == []:  # if sampling the objects
        objects = random.sample(sample_source, num_agents)
    elif sample_source is not None and surfaces == []:  # if sampling the surfaces
        surfaces = random.sample(sample_source, num_agents)
    script = " | ".join([f"<char{agent_id}> [{action}]" + (f" <{objects[agent_id][1]}> ({objects[agent_id][0]})" if objects is not None else "") + (f" <{surfaces[agent_id][1]}> ({surfaces[agent_id][0]})" if surfaces is not None else "") for agent_id in range(num_agents)])
    print("Running script:", script)
    while True:  # keep trying in case there are connection errors
        try:  # try running the script
            success, message = comm.render_script([script], recording=True, find_solution=True, processing_time_limit=10000, frame_rate=10, camera_mode=["PERSON_FROM_BACK"])
            print(f"Script Complete: {success}, {message}")
            break
        except (UnityEngineException, UnityCommunicationException) as e:  # the engine or communication can fail
            print(f"Script Fail: {e.text}")
    if surfaces is None:
        return objects
    elif objects is None:
        return surfaces
    return objects, surfaces

# sample two items, choose items that are not close together so the agents don't get stuck 
def __sample_objects__(sample_source:list, num:int=2):
    # bit of a hack, just resample until we get objects that aren't very close to each other
    # to optimize: start with a sample, and then resample objects that are close to each other
    while True:
        res = random.sample(sample_source)  # generate original sample, without replacement
        obj_ids = [x[0] for x in res]  # get the object IDs
        if len([True for edge in g["edges"] if edge["relation_type"] == "FACING" and "from" in obj_ids and "to" in obj_ids]) == 0:  # check edges to see if any objects are close
            break

# run the agents
if __name__ == "__main__":
    agents = threading.Thread(target=move_agents, args=(30,))
    agents.start()
    agents.join()

print("Done!")