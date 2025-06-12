# til-25-finals
Congratulations on making it through to the BrainHack TIL-AI 2025 Qualifiers!

Your autonomous reconnaissance bots are now operationally ready. As they navigate through the RL environment, when they encounter special missions, they will initiate your other mission tasks (e.g. ASR, CV, OCR) to be processed in the background and returned to HQ when a result is found. This is facilitated by your model orchestration server in `finals`, which should call your other model containers appropriately then pass the result back to our competition server. To let you test whether your solution works end-to-end, we have provided you a version of our competition server (HQ) for you to use when testing everything on your GCP instance.

There are also some slight changes to the `til-25-environment`, and it is recommended you ensure that your existing RL agents work (and possibly are re-trained) with the changes to the environment. Most notably:
- A bug has been fixed in the spawning locations of the agents. They now correctly alternate spawning across each of the corners of the map, rather than just the top left as a Scout and a subset of the rest of the locations.
- An infinite loop in arena generation has been patched.
- A flag has been exposed in the `info` dictionary returned to indicate whether a special mission has been initiated.
- The `step` observation component has been modified to have a maximum value of `NUM_ITERS+1` instead of `NUM_ITERS` to reduce the likelihood of off-by-one errors, and is now also manually cast to a `np.uint8`.

## Setup
Init and update all submodules (`til-25-environment` within the `test_competition_server` directory).

```Bash
git submodule update --init
```

Create a new `.env` file copied from the `.env.example` file. Also create a directory called `artifacts` to store testing artifacts from Docker so you can review them later:

```Bash
cp .env.example .env
mkdir -p artifacts
```

Also make sure that your data directory (either `novice` or `advanced`) is mounted in your home directory. If it's not, you should be able to mount it with the following:

```Bash
mkdir -p $HOME/$TRACK && sudo mount $HOME/$TRACK
```

## Submitting for finals
Submit the final versions of all your models using the `til submit` command, using the naming convention `{TEAM_NAME}-{TASK}:finals`. For example, if your team name is `team-ryan` and you're submitting your `asr` model for finals, the name would be `team-ryan-asr:finals`.

```Bash
til submit team-ryan-asr:finals
til submit team-ryan-cv:finals
til submit team-ryan-ocr:finals
til submit team-ryan-rl:finals
```

## Run
To test everything working together, run the `test.sh` script, which will:
1. Build your model orchestration server from the `finals/` directory, and push it to Artifact Registry.
2. Run `docker compose up`, testing the full setup end-to-end with a mock competition server.

```Bash
bash test.sh
```

It should take a couple of minutes to run because it waits for 2 seconds for all the RL results to return. If you know your RL agent runs substantially faster than that (as the overwhelming majority of the submitted RL agents do), then feel free to modify the `RL_TIME_CUTOFF` value in `src/constants.py` so your test runs faster. 

If everything works without errors, hooray! We'll see you at the IRL finals at MBS on June 11th and 12th <3