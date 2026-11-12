"""Models that run.py can train. Each module here is one model, selected with `run.py --model <name>`.

A model module provides:

    train(env, episodes, schedule, recorder, out_dir) -> list of per-episode returns

    env        gymnasium env (TimeLimit-wrapped, continuous actions)
    episodes   number of training episodes
    schedule   call schedule(env.unwrapped, episode) before each env.reset(); it changes the task
               (weakened engine, good pad, ...) and returns a message when something changed, else None
    recorder   episode_recorder.EpisodeRecorder: call recorder.start(env, episode) after env.reset(),
               recorder.step(env, reward) after each env.step(), recorder.end(env, ret, terminated, truncated)
               when the episode is over
    out_dir    folder for checkpoints
"""
