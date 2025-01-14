# cd into virtualhome repo
from virtualhome.virtualhome.simulation.unity_simulator import comm_unity

YOUR_FILE_NAME = "./macos_exec.v2.3.0.app" # Your path to the simulator
port= "8080" # or your preferred port

comm = comm_unity.UnityCommunication(
    file_name=YOUR_FILE_NAME,
    port=port
)

env_id = 0 # env_id ranges from 0 to 6
comm.reset(env_id)