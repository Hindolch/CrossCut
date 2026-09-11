# Diamond obtained: 137 steps

One GPT-6 Astra agent controlled unmodified Python Crafter 1.8.3 from the local Codex task. The game ran detached on the supplied Jarvis instance with an NVIDIA L4. Crafter's simulation runs on CPU; Astra inference stayed in Codex. No training or pretrained gameplay policy was used.

The first episode (seed 42) obtained one diamond after **137 primitive actions**, out of a lifetime budget of **1,000,000**. The native `collect_diamond` achievement and inventory both equal **1**. There were no deaths or resets. The controller stopped on success.

[W&B run](https://wandb.ai/nileshsarkar-ai/crosscut-diamonds/runs/diamonds) contains the live metrics, final outcome, complete video, and a gameplay artifact with this evidence bundle. Git automatically committed and pushed achievement milestones during play; revision 15 records the winning state in commit `90e63cc`.

## Recording

- **[gameplay.mp4](gameplay.mp4)**: complete video in a browser-compatible format, enlarged to 768 x 768 with nearest-neighbor scaling.
- **[full-session.mp4](full-session.mp4)**: original lossless RGB video, 256 x 256.
- **[full-recording.tar.gz](full-recording.tar.gz)**: all original PNGs and per-frame state/action metadata, committed server journal, automatic export outputs, and server/finalizer logs. Extract it to find frames in `data/server/recordings/diamonds-astra-1/episode-000001/`.
- **[diamonds-astra-1-episode-000001.jsonl](diamonds-astra-1-episode-000001.jsonl)**: complete per-frame action and game-state log.
- **[final-state.json](final-state.json)** and **[winning-frame.png](winning-frame.png)**: final native state and observation.
- **[conversion-status.json](conversion-status.json)** and **[diamonds-astra-1-manifest.json](diamonds-astra-1-manifest.json)**: detached finalizer evidence. Absolute paths describe their original locations on Jarvis.
- **[verification.json](verification.json)**: verification of the finished run. Every decoded lossless video frame matches its original PNG pixel-for-pixel, every PNG hash matches its metadata, and all actions match the committed journal.

Both videos contain **138 frames**: the initial observation plus every one of the 137 action results. At 10 FPS, playback lasts **13.8 seconds**. This is presentation time; model thinking and network delays are omitted, with no game frames omitted.

The action interface exposed the current observation, inventory, achievements, and visible local tile map. No hidden world map or game-state modifications were used. The original game ran at the normal difficulty. The raw recording and lossless master preserve the evidence independently of W&B.

API credentials and temporary SSH private keys are excluded from this repository and recording bundle.
