import gc
import numpy as np
import os
import subprocess
import torch
import shutil
import json
import time
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, PreTrainedTokenizer, PreTrainedTokenizerFast, GenerationConfig, AutoTokenizer, LlamaForCausalLM
from tkinter.filedialog import askdirectory, askopenfilename
from colorama import init, Fore, Style

from peft import PeftModel

import torch.nn as nn

model_path1 = '/home/kasm-user/Desktop/google-gemma-2-9b'
lora_path = '/home/kasm-user/Desktop/google-gemma-2-9b/lora/testinglora'
save_path = '/home/kasm-user/Desktop/google-gemma-2-9b/lora/testinglora/out'

max_shard_size = "4000MB"  # Output shard size

print("Starting script, please wait...")        

class NoInit:
    def __enter__(self):
        def noop(*args, **kwargs):
            pass

        (k, u, n) = (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        )
        torch.nn.init.kaiming_uniform_ = noop
        torch.nn.init.uniform_ = noop
        torch.nn.init.normal_ = noop

        transformers.modeling_utils._init_weights = False
        self.funcs = (k, u, n)

    def __exit__(self, *args):
        (k, u, n) = self.funcs
        (
            torch.nn.init.kaiming_uniform_,
            torch.nn.init.uniform_,
            torch.nn.init.normal_,
        ) = (
            k,
            u,
            n,
        )
        transformers.modeling_utils._init_weights = True

with torch.no_grad(): 
    
    with NoInit():
        torch.set_default_dtype(torch.float32)
    
        device = torch.device("cpu")
        print(device)
    
        # Model 1    
        print("Loading Model 1 (" + model_path1 + ")...")
        model1 = AutoModelForCausalLM.from_pretrained(model_path1, torch_dtype=torch.float32) 
        model1 = model1.to(device)
        model1.eval()
        print("Model 1 Loaded. Dtype: " + str(model1.dtype))
        
        #Lora
        print("Loading LoRa (" + lora_path + ")...")
        model1 = PeftModel.from_pretrained(model1, lora_path, torch_dtype=torch.float32)
        model1 = model1.merge_and_unload()
        
        tokenizer = AutoTokenizer.from_pretrained(model_path1)
        
        print("Saving merged model (" + save_path + ")...")
        model1.to(dtype=torch.bfloat16)
        model1.save_pretrained(save_path, max_shard_size=max_shard_size, safe_serialization=True, progressbar=True)
        tokenizer.save_pretrained(save_path)
                
                
                
            
            

        
        
