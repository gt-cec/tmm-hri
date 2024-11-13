# remove the pkls that already exist
rm episodes/episode_2024-09-04-16-32_agents_2_run_19/1/Action_*.pkl

# generate the new ones
python3.12 preprocess_sim_detection.py
