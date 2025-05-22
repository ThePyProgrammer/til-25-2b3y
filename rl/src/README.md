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
    - [ ] make pruning less strict, ignore route.contains

## testing
```
python demo.py --control 0,1,2,3 --scout_target 15,15 --seed 720354 --human
```
