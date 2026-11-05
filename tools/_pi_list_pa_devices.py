import sounddevice as sd

for i, d in enumerate(sd.query_devices()):
    print(
        f"{i:3d} in={d['max_input_channels']:2d} out={d['max_output_channels']:2d} "
        f"sr={int(d['default_samplerate']):5d}  {d['name']}"
    )
print("default", sd.default.device)
