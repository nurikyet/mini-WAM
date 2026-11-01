# Pinned upstream code

This repository includes the actual upstream NanoWM backbone and XLand-MiniGrid
simulator as ordinary files. There are no nested Git repositories or submodules.
`SOURCE.json` records every included file's SHA256 and its upstream commit.

* NanoWM: [simchowitzlabpublic/nano-world-model, cc1f151](https://github.com/simchowitzlabpublic/nano-world-model/tree/cc1f151a9d8c7d7afd0cdc5de6a9ae220a7c90c0).
  Only `src/models/nanowm.py` is required by this adaptation. Unmodified MIT source.
* XLand-MiniGrid: [dunnolab/xland-minigrid, 0d306d1](https://github.com/dunnolab/xland-minigrid/tree/0d306d1414dcb6e451358d94e7d9a91879462ee1).
  Unmodified Apache-2.0 simulator source under `xminigrid/`. The upstream
  baseline trainer and its local Python 3.11 patch are not included or used;
  state PPO is implemented in `miniwam/server/expert.py`.

Their licenses are preserved in their respective directories. The custom WAM
action head, training loop, data adapter, and visual planner are in
`miniwam/server/`. They are an explicit extension of NanoWM, not the unpublished
summer-school implementation or a claim to reproduce that report.
