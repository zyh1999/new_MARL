# copy from https://github.com/deepmind/pysc2/issues/288
import mpyq
import six
import json


def get_replay_version(replay_path):
	with open(replay_path, "rb") as f:
		replay_data = f.read()
		replay_io = six.BytesIO()
		replay_io.write(replay_data)
		replay_io.seek(0)
		archive = mpyq.MPQArchive(replay_io).extract()
		metadata = json.loads(archive[b"replay.gamemetadata.json"].decode("utf-8"))
		print(metadata)


get_replay_version(r"/Applications/StarCraft II/Replays/5m_vs_6m_2021-12-07-09-56-54.SC2Replay")

# 1. get replay version
# 2. subl /Users/liushunyu/anaconda3/envs/marl/lib/python3.8/site-packages/pysc2/run_configs/lib.py
# 3. add new Version to VERSIONS
