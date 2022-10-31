# -*- coding: utf-8 -*-
"""
Endpoint Latency

@author: Jin Sakuma
"""

import argparse
import numpy as np
import pandas as pd
import wave
import time
import torch
import soundfile as sf
import textgrid
import glob
import os
from tqdm import tqdm


def get_recognize_frame(data, speech2text, mode='non_parallel'):
    output_list = []
    speech = data#.astype(np.float16)/32767.0 #32767 is the upper limit of 16-bit binary numbers and is used for the normalization of int to float.
    #sim_chunk_length =  800 # 640
    
    size = 4 # 20
    kernel_2, stride_2 = 3, 2  # sub = 4
    sim_chunk_length = (((size + 2) * 2) + (kernel_2 - 1) * stride_2) * 128

    if sim_chunk_length > 0:
        for i in range(len(speech)//sim_chunk_length):
            hyps = speech2text.streaming_decode(speech=speech[i*sim_chunk_length:(i+1)*sim_chunk_length], is_final=False)
            if mode == 'parallel':
                hyps = hyps[1]
            results = speech2text.hypotheses_to_results(hyps)                            
            if results is not None and len(results) > 0:
                nbests = [text for text, token, token_int, hyp in results]
                text = nbests[0] if nbests is not None and len(nbests) > 0 else ""
                output_list.append(text)
            else:
                output_list.append("")

        hyps = speech2text.streaming_decode(speech[(i+1)*sim_chunk_length:len(speech)], is_final=True)
        if mode == 'parallel':
                hyps = hyps[1]
        results = speech2text.hypotheses_to_results(hyps)
    else:
        hyps = speech2text.streaming_decode(speech, is_final=True)
        if mode == 'parallel':
                hyps = hyps[1]
        results = speech2text.hypotheses_to_results(hyps)        
    
    #nbests = [text for text, token, token_int, hyp in results]
    if results is not None and len(results) > 0:
        nbests = [text for text, token, token_int, hyp in results]
        text = nbests[0] if nbests is not None and len(nbests) > 0 else ""        
        output_list.append(text)
    else:
        output_list.append("")
        
    return output_list

def run(args, speech2text, outdir, split='test_eval92'):
    
    size = 4
    kernel_2, stride_2 = 3, 2  # sub = 4
    sim_chunk_length = (((size + 2) * 2) + (kernel_2 - 1) * stride_2) * 128
        
    with open('dump/raw/test_{}/wav.scp'.format(split), 'r') as f:
        lines = f.readlines()
        
    paths = []
    names = []
    for line in lines:
        name, path = line.split()
        if '.flac' in path:
            names.append(name)
            paths.append(path)
            
    with open('dump/raw/test_{}/text'.format(split), 'r') as f:
        lines = f.readlines()

    transcripts = []
    for i, line in enumerate(lines):
        splits = line.split()
        name = splits[0]
        text = splits[1:]
        assert name == names[i], 'file is not much'
        if '.flac' in path:
            transcripts.append(text)
       
    results = []
    texts = []
    is_ok = []
    for i in tqdm(range(len(paths))):
        name = names[i]
        wav_path = paths[i]
        tg_path = os.path.join(TEXTGRID , split, "{}.TextGrid".format(name))
        
        wav, samplerate = sf.read(wav_path)
        tg = textgrid.TextGrid.fromFile(tg_path)
        
        text = get_recognize_frame(wav, speech2text, args.mode)
        texts.append(text)
        
        last = ''
        idx = 0
        for i, t in enumerate(text):
            if len(t)>0 and last != t and t != '':
                last = t
                idx = i+1
            
        mint = None
        maxt = None
        for itv in tg[0]:
            wp = itv.mark
            #if wp == last.lower():
            if wp.lower() != '' and wp is not None:
                mint = itv.minTime
                maxt = itv.maxTime                    

        if last != '' and last != '[NOISE]' and maxt is not None:        
            ep = sim_chunk_length / 16000 * idx
            latency = (ep - maxt) * 1000
            results.append(latency)
            is_ok.append(True)
        else:
            results.append(None)
            is_ok.append(False)
            
        
    results = np.array(results)
    df = pd.DataFrame({"name": names,
                       "path": paths,
#                        "ref": transcripts,
#                        "hyp": texts,
                       "latency": results,
                       "flg": is_ok
                      })
        
    df.to_csv(os.path.join(outdir, 'latency_{}.csv'.format(split)), index=False)
    latency_arr = df['latency'].values
    print('EP50: {:.0f}ms, EP90: {:.0f}ms'.format(np.percentile(latency_arr, 50), np.percentile(latency_arr, 90)))    
    
                
TEXTGRID = './alignments'
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='asr_train_asr_cbs_transducer_8_4_4_baseline_raw_en_bpe80', help='exp name')
    parser.add_argument('--gpuid', type=int, default=-1, help='gpu id')
    parser.add_argument('--beamsize', type=int, default=1, help='beam size')
    parser.add_argument('--mode', type=str, default="non_parallel", help='decoding mode', choices=('non_parallel', 'parallel'),)
    args = parser.parse_args()            
       
    exp = args.exp
    if args.gpuid >= 0:
        device = 'cuda:{}'.format(args.gpuid)
    else:
        device = 'cpu'
    
    outdir = './exp/{}'.format(exp)    
    os.makedirs(outdir, exist_ok=True)
    
    if args.mode == 'parallel':
        from espnet2.bin.asr_parallel_transducer_inference import Speech2Text
    else:
        from espnet2.bin.asr_transducer_inference import Speech2Text
        
    speech2text = Speech2Text(
        asr_train_config="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/wsj/asr1/exp/{}/config.yaml".format(exp),
        asr_model_file="/mnt/aoni04/jsakuma/development/espnet-g05-1.8/egs2/wsj/asr1/exp/{}/valid.loss_transducer.ave_10best.pth".format(exp),
        token_type=None,
        bpemodel=None,
        beam_search_config={"search_type": "greedy"},
        beam_size=args.beamsize,
        lm_weight=0.0,
        nbest=1,
        device = device,
        streaming = True,
    )
        
    splits = ['eval92', 'dev93']
    #splits = ['eval92']
    for split in splits:
        run(args, speech2text, outdir, split)
        