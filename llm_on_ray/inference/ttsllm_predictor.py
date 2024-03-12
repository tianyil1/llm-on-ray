import os
import sys
import re
import torch
import numpy as np
np.set_printoptions(threshold=np.inf)

from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

from llm_on_ray.inference.inference_config import InferenceConfig, GenerateResult
from llm_on_ray.inference.transformer_predictor import TransformerPredictor

class TTSllmPredictor(TransformerPredictor):
    def __init__(self, infer_conf: InferenceConfig):
        self.device = torch.device(infer_conf.device)
        self.model_desc = infer_conf.model_description
        model_config = self.model_desc.config
        model = SpeechT5ForTextToSpeech.from_pretrained(
            self.model_desc.model_id_or_path
        )
        processor = SpeechT5Processor.from_pretrained(self.model_desc.model_id_or_path)
        vocoder = SpeechT5HifiGan.from_pretrained(self.model_desc.vcoder_name_or_path)
        self.speaker_embeddings = {
        "BDL": "cmu_us_bdl_arctic-wav-arctic_a0009.npy",
        "CLB": "cmu_us_clb_arctic-wav-arctic_a0144.npy",
        "KSP": "cmu_us_ksp_arctic-wav-arctic_b0087.npy",
        "RMS": "cmu_us_rms_arctic-wav-arctic_b0353.npy",
        "SLT": "cmu_us_slt_arctic-wav-arctic_a0508.npy",
        }

        model = model.eval()
        self.model = model
        self.processor = processor
        self.vocoder = vocoder
        

    def generate(self, prompt, **config):
        print(prompt)
        prompt = prompt[0]
        self._process_config(config)
        vocies = re.search(r"voices: ([^\,]*)", prompt).group(1)
        text = re.search(r"text: (.*)", prompt).group(1)
        if vocies in self.speaker_embeddings:
            speaker_embedding = self.speaker_embeddings[vocies]
        else:
            raise Warning(f"Speaker embedding {vocies} not found.")
            speaker_embedding = self.speaker_embeddings["BDL"]
        speaker_embedding = torch.tensor(np.load(os.path.join(self.model_desc.speaker_embedding_path, speaker_embedding))).unsqueeze(0)

        if len(prompt.strip()) == 0:
            return np.array_str((16000, np.zeros(0).astype(np.int16)))
        
        inputs = self.processor(text=text, return_tensors="pt")

        # limit input length
        input_ids = inputs["input_ids"]
        input_ids = input_ids[..., :self.model.config.max_text_positions]

        speech = self.model.generate_speech(input_ids, speaker_embedding, vocoder=self.vocoder)
        print(speech.numpy().shape)
        speech = [np.array_str((speech.numpy() * 32767).astype(np.int16), max_line_width=np.inf)]
        return GenerateResult(text=speech)


        
