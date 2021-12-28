import pandas as pd
import numpy as np
import plotly.graph_objects as go
import kaleido
def get_signal_basecall_for_plotting(df,readid):
    idx=df.loc[df['read_id']==readid].index.values[0]
    signal=df.loc[idx]['signal']
    move=df.loc[idx]['move']
    trace=df.loc[idx]['trace']
    seq_fastq=df.loc[idx]['seq']
    stride=df.loc[idx]['stride']
    first_sample_template=df.loc[idx]['first_sample_template']

    basecall_sequence = []
    pivot = first_sample_template
    cur_base_idx = 0
    for i in range(1, len(move)):
        if move[i] == 1:
            cur_move = {}
            cur_move["base"] = seq_fastq[cur_base_idx]
            cur_base_idx += 1
            cur_move["start"] = pivot
            cur_move["end"] = first_sample_template + i * stride
            pivot = cur_move["end"]
            basecall_sequence.append(cur_move)
    last_move = {}
    last_move["base"] = seq_fastq[cur_base_idx]
    last_move["start"] = pivot
    last_move["end"] = first_sample_template + len(move) * stride
    basecall_sequence.append(last_move)
    #print(f'read of length {len(basecall_sequence)} bases')
    return signal, basecall_sequence

def plot_basecalling(read_id, basecall_sequence, signal, save_path=None, mod=''):
    fig = go.Figure([go.Scatter(y=signal)])
    fig.update_layout(title=mod+read_id)
    colors = {"A": "green", "U": "red", "C": "violet", "G": "yellow"}
    for i, cur_base in enumerate(basecall_sequence):
        if i%100 ==0 :
            print(f'base {i}')
            #print(f'RAM memory % used: {psutil.virtual_memory()[2]}')
        fig.add_vrect(x0=cur_base["start"], x1=cur_base["end"], 
                  annotation_text=cur_base["base"], annotation_position="top left",
                  fillcolor=colors[cur_base["base"]], opacity=0.25, line_width=0)
    if save_path is not None:
        fig.write_image(save_path)
    fig.show()
    for cur_base in basecall_sequence:
        print(cur_base["base"], end="")