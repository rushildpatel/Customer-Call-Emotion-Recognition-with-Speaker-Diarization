import numpy as np
import pandas as pd


from pyannote.audio import Pipeline

pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token="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")

# apply the pipeline to an audio file
diarization = pipeline("D:/SPIT/8th Sem/Major Project/SpeakerDiarization/audio/test_bro_new.wav", num_speakers = 2)

# dump the diarization output to disk using RTTM format
with open("D:/SPIT/8th Sem/Major Project/SpeakerDiarization/audio/rttm/test_bro_new.rttm", "w") as rttm:
    diarization.write_rttm(rttm)