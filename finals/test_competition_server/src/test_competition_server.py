import asyncio
import json
import logging
import os
import traceback
from pathlib import Path
from time import time

import constants
import imageio
import numpy as np
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from task_handler import TaskHandler
from til_environment.gridworld import Action, parallel_env
from websockets.exceptions import ConnectionClosed

# Default action, STAY
DEFAULT_ACTION = Action.STAY.value

logger = logging.getLogger("uvicorn.error")
logger.setLevel(logging.INFO)

track: str = os.environ["TEAM_TRACK"]

# Filepath to load all data from
data_dir = Path("../data")

app = FastAPI()
app.mount(
    "/data",
    StaticFiles(directory=data_dir.resolve()),
    name="data",
)


@app.get("/health")
async def health():
    return "OK"


# Websocket connection manager
class ConnectionManager:
    in_progress = False

    def __init__(self):
        self.auto_step = True
        self.round = 0
        self.step_num = 0
        self.team_names = [os.environ["TEAM_NAME"], "team-2", "team-3", "team-4"]
        self.team_agent_mapping = {}
        self.agent_team_mapping = {}
        self.observations = None

        self.match_start_time = time()
        self.match_out_dir = f"../artifacts/match_{self.match_start_time}"
        os.makedirs(self.match_out_dir, exist_ok=True)

        for team_name, idx in zip(self.team_names, range(len(self.team_names))):
            agent_id = f"player_{idx}"
            self.team_agent_mapping[team_name] = agent_id
            self.agent_team_mapping[agent_id] = team_name

        self.env = parallel_env(
            env_wrappers=[], render_mode="rgb_array", novice=track == "novice"
        )
        self.frames: list[np.ndarray] = []
        observations, _ = self.env.reset()
        self.frames.append(self.env.render())
        self.observations = observations

        self.team_connections: dict[str, WebSocket | None] = {
            name: None for name in self.team_names
        }
        self.match_results = {
            "teams": self.team_names,
            "num_rounds": constants.NUM_ROUNDS,
            "track": track,
            "rounds": [
                {"round": i, "steps": [], "scout_results": []}
                for i in range(constants.NUM_ROUNDS)
            ],
        }
        # Init step-specific variables
        self.set_default_actions()
        self.start_times = {team: 0 for team in self.team_names}
        self.task_start_time = 0

    def update_task(self):
        return task_handler.get_task_data()

    async def team_connect(self, websocket: WebSocket, team_name: str):
        if team_name not in self.team_names:
            await websocket.close(reason=f"Invalid team {team_name}")
        elif self.team_connections[team_name] == None:
            await websocket.accept()
            self.team_connections[team_name] = websocket
        else:
            logger.info(self.team_connections)
            try:
                await self.team_connections[team_name].send_json({"type": "health"})
                _ = await self.team_connections[team_name].receive()
            except (WebSocketDisconnect, ConnectionClosed, RuntimeError):
                await self.team_disconnect(team_name)
                await websocket.accept()
                self.team_connections[team_name] = websocket
            else:
                await websocket.close(
                    reason=f"There is already a team connected with name {team_name}!"
                )
        # Print which teams are connected
        logger.info(self.team_connections)

    async def team_disconnect(self, team_name: str, message: str = "Disconnected"):
        try:
            await self.team_connections[team_name].close(message)
        except:
            pass
        self.team_connections[team_name] = None

    async def step_team(
        self,
        team_name: str,
        websocket: WebSocket,
        observation: dict,
    ):
        observation = {
            k: v if type(v) is int else v.tolist() for k, v in observation.items()
        }
        await websocket.send_json(
            {"type": "task", "task": "rl", "observation": observation}
        )
        self.start_times[team_name] = time()

    async def broadcast_teams(self, message: dict):
        return await asyncio.gather(
            *[
                connection.send_json(message)
                for connection in self.team_connections.values()
                if connection
            ],
            return_exceptions=True,
        )

    async def run_until_stop(self):
        if self.round >= constants.NUM_ROUNDS:
            self.in_progress = False
            return
        self.in_progress = True
        while self.auto_step and self.round < constants.NUM_ROUNDS:
            logger.debug(f"running round {self.round} step {self.step_num}")
            await self.step()
        else:
            if self.round >= constants.NUM_ROUNDS:
                await self.broadcast_teams({"type": "done"})
        self.in_progress = False

    def set_default_actions(self):
        self.actions = {
            # NOTE: to make the other agents actually do something, replace this line with
            # agent: self.env.aec_env.action_space(agent).sample()
            # or some other agent control code
            agent: DEFAULT_ACTION
            for agent in self.team_agent_mapping.values()
        }

    # step
    async def step(self):
        try:
            start_time = time()
            self.set_default_actions()
            _tasks = [
                self.step_team(
                    team,
                    connection,
                    self.observations[self.team_agent_mapping[team]],
                )
                for team, connection in self.team_connections.items()
                if connection
            ]
            await asyncio.gather(*_tasks, return_exceptions=True)
            # Sleep until constants.RL_TIME_CUTOFF sec has passed
            await asyncio.sleep(constants.RL_TIME_CUTOFF - time() + start_time)
            logger.debug(f"completed in {time() - start_time:.2f}s")
            # Copy updated actions
            _actions = self.actions.copy()
            # Update rewards
            observations, rewards, terminations, truncations, infos = self.env.step(
                _actions
            )
            step_end_tasks = []
            if any([info["add_mission"] for info in infos.values()]):
                to_send = len(task_handler.queue) == 0 and task_handler.can_get_new
                # Add additional items to task queue
                task_handler.add_tasks(constants.QUEUE_ITEMS_PER_MISSION)
                if to_send:
                    try:
                        logger.info("sending to Scout")
                        # Send to the Scout
                        connection = self.team_connections[
                            self.agent_team_mapping[self.env.aec_env.scout]
                        ]
                        task_data = manager.update_task()
                        if connection is not None and task_data is not None:
                            manager.task_start_time = time()
                            step_end_tasks.append(connection.send_json(task_data))
                    except Exception as err:
                        logger.error(
                            "Error occurred while queueing send task:",
                            traceback.format_exc(),
                        )
            self.observations = observations
            self.match_results["rounds"][self.round]["steps"].append(
                {
                    "actions": {
                        k: v if type(v) is int else v.tolist()
                        for k, v in _actions.items()
                    },
                    "rewards": rewards,
                }
            )
            if any(terminations.values()) or any(truncations.values()):
                logger.info(f"done with round {self.round}")
                # Save video of round
                self.frames.append(self.env.render())
                imageio.mimsave(
                    f"{self.match_out_dir}/round_{self.round}.mp4", self.frames, fps=20
                )
                self.frames.clear()

                with open(
                    f"{self.match_out_dir}/match_results.json", "w"
                ) as results_file:
                    json.dump(manager.match_results, results_file)

                # Prepare for next round
                self.round += 1
                self.step_num = 0
                observations, _ = self.env.reset()
                self.frames.append(self.env.render())
                self.observations = observations
                task_handler.reset()
            else:
                self.step_num += 1
                self.frames.append(self.env.render())
            await asyncio.gather(*step_end_tasks)
        except Exception as e:
            logger.exception(e)
            self.auto_step = False


manager = ConnectionManager()
task_handler = TaskHandler(data_dir)


@app.post("/start")
async def start(background_tasks: BackgroundTasks):
    manager.auto_step = True
    if not manager.in_progress:
        background_tasks.add_task(manager.run_until_stop)


@app.post("/stop")
async def stop():
    manager.auto_step = False


@app.websocket("/ws/{team_name}")
async def team_endpoint(websocket: WebSocket, team_name: str):
    await manager.team_connect(websocket, team_name)
    try:
        while True:
            data = await websocket.receive_json()
            if data["task"] == "rl":
                elapsed = time() - manager.start_times[team_name]
                logger.debug(f"rl elapsed for {team_name}: {elapsed}")
                logger.debug(data)
                try:
                    # Reject invalid RL returns
                    if manager.step_num != data["result"]["step"]:
                        logger.info(
                            f"Rejecting RL data {data} for {team_name}: it should be for step {manager.step_num}"
                        )
                        continue
                    if _act := Action(data["result"]["action"]):
                        manager.actions[manager.team_agent_mapping[team_name]] = (
                            _act.value
                        )
                    continue
                except ValueError:
                    logger.info(
                        f"Rejecting RL data {data} for {team_name}: invalid action"
                    )
                    continue
            elapsed = time() - manager.task_start_time
            logger.debug(f"elapsed: {elapsed}")

            # Check if this team is meant to be the Scout
            if manager.env.aec_env.scout != manager.team_agent_mapping[team_name]:
                logger.info(f"Rejecting {data} for team {team_name}: not the Scout!")
                continue

            score = task_handler.eval_task_result(data, elapsed)
            manager.match_results["rounds"][manager.round]["scout_results"].append(
                {
                    "data": data,
                    "score": score,
                }
            )
            logger.info(
                f"Team {team_name} achieved score {score} for task type {data['task']}"
            )

            # Send next task
            task_data = manager.update_task()
            if task_data is not None:
                manager.task_start_time = time()
                await websocket.send_json(task_data)
    except (WebSocketDisconnect, ConnectionClosed):
        logger.info(f"Team '{team_name}' disconnected")
    finally:
        await manager.team_disconnect(team_name)
