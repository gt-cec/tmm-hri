# sim_start.py: launch the simulator

from virtualhome.virtualhome.simulation.unity_simulator import comm_unity

file_name = "./linux_exec/linux_exec.v2.3.0.x86_64"
port= "5050"

comm = comm_unity.UnityCommunication(
    file_name = file_name,
    port = port,
    graphics = False
    # x_display=":0"
)

env_id = 0 # env_id ranges from 0 to 6
comm.reset(env_id)