## todo

- [x] create a proper replay buffer (autoregressive style?)
- [ ] probably rewrite the trainer, dqn imple seems ok?
- [ ] lstm cached hidden + cell states @ inference time
- reward modelling:
    - [ ] exploration reward for guards
    - [ ] guard visibility penalty for scout, scout visibility reward for guards
- ideas to experiment with:
    - [ ] create our own record of the map to path find towards/away from the scout/guards
    - [ ] teacher (perfect knowledge) to student (limited vision)
    - [ ] "league" of networks (alphastar? style)
    - [x] store our own map of the maze

- who even needs rl?:
    - [x] the starting location of the scout is always (0, 0),
        so the guards could totally just pathfind to some region of interest where the scout could be
    - [ ] make the guard face in the direction which can eliminate the most probas ie.
    instead of favouring regions of high proba, favour regions which the most trajectories pass through
    - [x] the current pruning makes the mistake of delete all including edge trajectories which are needed
