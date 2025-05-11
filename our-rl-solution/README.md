## todo

- [x] create a proper replay buffer (autoregressive style?)
- [ ] probably rewrite the trainer, dqn imple seems ok?
- [ ] lstm cached hidden + cell states @ inference time
- reward modelling:
    - [ ] exploration reward for guards
    - [ ] guard penalty for scout, scout reward for guards
    - [ ] shared rewards for (+ for scout, - for guards and vice versa)
- ideas to experiment with:
    - [ ] create our own record of the map to path find towards/away from the scout/guards
    - [ ] teacher (perfect knowledge) to student (limited vision)
    - [ ] "league" of networks (alphastar? style)
