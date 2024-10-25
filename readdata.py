import ffmpeg
url = 'https://pull-flv-l11.douyincdn.com/stage/stream-115536659836043409_sd5.flv?expire=1722236858&sign=852fcd1e2141a40a98d9d64370c9310c&abr_pts=-800&_session_id=037-2024072215073791C811E464A905254065.1721632058846.59755'
ffmpeg_process = (
        ffmpeg
        .input(url)
        .output('pipe:', format='wav', acodec="pcm_s16le", ar=16000, ac=1)
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
    )
print("ffmpeg -i {} -ac 1 -ar 16000 data.wav".format(url))
print("aaaaaaaaaaaa")
print(ffmpeg_process.stdout.read(1024))