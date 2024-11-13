import ffmpeg

url="https://alimov2.a.kwimgs.com/upic/2024/11/12/10/BMjAyNDExMTIxMDI1NDJfMzM4Mjk1ODQ2OV8xNDg0MDk0NTEwMzhfMl8z_b_Bf727794a2d9e19c4419d98b78880558a.mp4?clientCacheKey=3xjccsifdk53re2_b.mp4&tt=b&di=249d7626&bp=1000"
ffmpeg_process = (
    ffmpeg
    .input(url).output('output.wav', format='wav', acodec="pcm_s16le", ar=16000, ac=1)
    .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
)
ffmpeg_process.wait()
print("ffmpeg -i {} -ac 1 -ar 16000 data.wav".format(url))
print("aaaaaaaaaaaa")
# print(ffmpeg_process.stdout.read(1024))
