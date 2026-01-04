from funasr import AutoModel
# paraformer-zh is a multi-functional asr model
# use vad, punc, spk or not as you need
model = AutoModel(model="paraformer-zh",  vad_model="fsmn-vad",  punc_model="ct-punc", 
                  # spk_model="cam++", 
                  )
res = model.generate(input=f"{model.model_path}/example/en.mp3", 
                     batch_size_s=300, 
                     hotword='紫东太初')
print(res)