# based on Shortest Path Follower example from github.com/facebookresearch/habitat-lab

import os
import shutil

import numpy as np

import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video
import subprocess

cv2 = try_cv2_import()

class SimpleRLEnv(habitat.RLEnv):
    def get_reward_range(self):
        return [-1, 1]

    def get_reward(self, observations):
        return 0

    def get_done(self, observations):
        return self.habitat_env.episode_over

    def get_info(self, observations):
        return self.habitat_env.get_metrics()


def draw_top_down_map(info, output_size):
    return maps.colorize_draw_agent_and_fit_to_height(
        info["top_down_map"], output_size
    )


def run_agents():
    config = habitat.get_config(
        config_path="hab_1_config.yaml",
        overrides=[
            "+habitat/task/measurements@habitat.task.measurements.top_down_map=top_down_map"
        ],
    )

    # compress the json file
    subprocess.run(["gzip", "hab_1_episodes.json", "--keep", "--force"])

    env = SimpleRLEnv(config=config) 

    goal_radius = env.episodes[0].goals[0].radius
    if goal_radius is None:
        goal_radius = config.habitat.simulator.forward_step_size

    robot = ShortestPathFollower(
        env.habitat_env.sim, goal_radius, False
    )

    # human = ShortestPathFollower(
    #     env.habitat_env.sim, goal_radius, False
    # )

    print("Environment creation successful")
    env.reset()

    dirname = "output"
    if os.path.exists(dirname):
        shutil.rmtree(dirname)
    os.makedirs(dirname)
    print("Agent stepping around inside environment.", env.current_episode())
    images = []
    while not env.habitat_env.episode_over:
        robot_best_action = robot.get_next_action(
            env.habitat_env.current_episode.goals[0].position
        )
        
        if robot_best_action is None:
            break

        observations, reward, done, info = env.step(robot_best_action)
        im = observations["rgb"]
        top_down_map = draw_top_down_map(info, im.shape[0])
        output_im = np.concatenate((im, top_down_map), axis=1)
        images.append(output_im)

    images_to_video(images, dirname, "jack-run")
    print("Episode finished")


def main():
    run_agents()


if __name__ == "__main__":
    main()