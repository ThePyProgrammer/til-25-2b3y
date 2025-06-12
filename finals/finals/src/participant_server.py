import asyncio
import json
import os
import traceback
from urllib.parse import quote

import websockets
from models_manager import ModelsManager

TEAM_NAME = os.environ["TEAM_NAME"]
LOCAL_IP = os.environ["LOCAL_IP"]
SERVER_IP = os.environ["COMPETITION_SERVER_IP"]
SERVER_PORT = os.environ["COMPETITION_SERVER_PORT"]

manager = ModelsManager(LOCAL_IP)


async def task_handler(data: dict) -> None:
    # parse data and send to model manager
    match data["task"]:
        case "asr":
            return await manager.run_asr(data["b64"])
        case "cv":
            return await manager.run_cv(data["b64"])
        case "ocr":
            return await manager.run_ocr(data["b64"])
        case "rl":
            # add step number to return value to make sure the RL action corresponds to the step
            action = await manager.run_rl(data["observation"])
            return {"step": data["observation"]["step"], "action": action}
        case "surprise":
            return await manager.run_surprise(data["slices"])

async def handle_task_and_send_result(websocket, data: dict):
    """Handle a task and send the result when complete"""
    try:
        result = await task_handler(data)
        await manager.send_result(websocket, {"task": data["task"], "result": result})
    except Exception as e:
        print(f"Error handling task {data.get('task', 'unknown')}: {e}")
        traceback.print_exception(e)


async def server():
    async for websocket in websockets.connect(
        quote(f"ws://{SERVER_IP}:{SERVER_PORT}/ws/{TEAM_NAME}", safe="/:"),
        max_size=2**24,
    ):
        print(f"connecting to competition server {SERVER_IP} at port {SERVER_PORT}")

        # Keep track of running tasks so we can clean them up if needed
        running_tasks: set[asyncio.Task] = set()

        try:
            while True:
                # Receive json data from server
                socket_input = await websocket.recv()
                if type(socket_input) is str:
                    data = json.loads(socket_input)
                    match data["type"]:
                        case "task":
                            # Create task and add to running tasks set
                            task = asyncio.create_task(
                                handle_task_and_send_result(websocket, data)
                            )
                            running_tasks.add(task)

                            # Remove completed tasks from the set to prevent memory leaks
                            task.add_done_callback(running_tasks.discard)

                        case "done":
                            # Handle done update
                            print("done!")

                            # Wait for all running tasks to complete before breaking
                            if running_tasks:
                                print(
                                    f"Waiting for {len(running_tasks)} tasks to complete..."
                                )
                                await asyncio.gather(
                                    *running_tasks, return_exceptions=True
                                )
                            await manager.exit()
                            break

                        case "health":
                            await manager.send_result(websocket, {"health": "ok"})

                        case _:
                            print(
                                f"received invalid text data of type {data['type']}:",
                                data,
                                sep="\n",
                            )
                else:
                    print(f"received invalid data of type {type(socket_input)}")

        except websockets.ConnectionClosed:
            # Cancel any running tasks when connection is lost
            await shutdown(running_tasks, manager)
            continue
        except KeyboardInterrupt:
            # Cancel running tasks on keyboard interrupt
            await shutdown(running_tasks, manager)
            break
        except Exception as e:
            traceback.print_exception(e)
            # Cancel running tasks on unexpected errors
            await shutdown(running_tasks, manager)
        else:
            break


async def shutdown(running_tasks: set[asyncio.Task], manager: ModelsManager):
    for task in running_tasks:
        task.cancel()
    return await manager.exit()


if __name__ == "__main__":
    asyncio.run(server())
