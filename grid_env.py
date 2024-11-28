from __future__ import annotations

import random
from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv
import copy


class GridEnv(MiniGridEnv):
    def __init__(
        self,
        size=10,
        seed=None,
        agent_start_pos=None,
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)

        self.agent_start_pos = (random.randint(1, size - 2), random.randint(1, size - 2))
        self.agent_start_dir = agent_start_dir
        self.goal_x = random.randint(1, size - 2)
        self.goal_y = random.randint(1, size - 2)

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=True,
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "grand mission"

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.put_obj(Goal(), self.goal_x, self.goal_y)
        self.agent_pos = self.agent_start_pos
        self.agent_dir = self.agent_start_dir

        self.mission = "grand mission"

    def get_state(self):
        return (self.agent_pos[0], self.agent_pos[1], self.agent_dir)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        state = self.get_state()
        return state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        return self.get_state()

    def copy(self):
        return copy.deepcopy(self)


def main():
    env = GridEnv(render_mode="human", seed=42, size=100)

    manual_control = ManualControl(env, seed=42)
    manual_control.start()

if __name__ == "__main__":
    main()
