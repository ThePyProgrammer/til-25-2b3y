## Optimizations

- lru caching improves by ~25%


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
    - [ ] fit a trajectory through every seen visited tile (one for each direction). connect from the starting point to that trajectory assuming most direct path is taken, start drawing trajectories from the endpoint, fast forwarding the steps given the budget of starting from starting point.
    - [ ] there is a big, the point on 0, 0 isn't collected when the scout spawns there, so I need to prevent there from destroying my trajectories


## testing
```
python demo.py --control 0,1,2,3 --scout_target 15,15 --seed 720354 --human
```
