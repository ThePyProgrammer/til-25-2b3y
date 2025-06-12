"""Runs the CV server."""

# Unless you want to do something special with the server, you shouldn't need
# to change anything in this file.


import base64
from typing import Any

from fastapi import FastAPI, Request

from cv_manager import CVManager, BATCH_SIZE
from more_itertools import batched


app = FastAPI()
manager = CVManager()


@app.post("/cv")
async def cv(request: Request) -> dict[str, list[list[dict[str, Any]]]]:
    """Performs CV object detection on image frames.

    Args:
        request: The API request. Contains a list of images, encoded in
            base-64.

    Returns:
        A `dict` with a single key, `"predictions"`, mapping to a `list` of
        `dict`s containing your CV model's predictions, in the same order as
        which appears in `request`. See `cv/README.md` for the expected format.
    """

    inputs_json = await request.json()

    predictions = []
    for instances in batched(inputs_json["instances"], BATCH_SIZE):
        batch_image_bytes = list(map(lambda instance: base64.b64decode(instance["b64"]), instances))

        # Performs object detection and appends the result.
        detections = manager.cv(batch_image_bytes)
        predictions.extend(detections)

#     for instance in inputs_json["instances"]:
#         batch_image_bytes = base64.b64decode(instance["b64"])

#         # Performs object detection and appends the result.
#         detections = manager.cv_slice(batch_image_bytes)
#         predictions.append(detections)

    return {"predictions": predictions}


@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint for your model."""
    return {"message": "health ok"}
