import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def plot_predicted_vs_actual_actions(output_json):
    df = (
        pd.DataFrame.from_dict(output_json, orient="index")
        .rename_axis("frame")
        .reset_index()
    )

    df["frame"] = df["frame"].astype(int)
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

    df = df.sort_values("frame")

    

    fig, axes = plt.subplots(2, 1, figsize=(12, 12), sharex=True)

    # ACTUAL EVENTS
    axes[0].plot(df["frame"], df["y"], lw=1, label="y-position")

    hits = df[df["action"] == "hit"]["frame"]
    bounces = df[df["action"] == "bounce"]["frame"]

    axes[0].vlines(
        hits,
        ymin=df["y"].min(),
        ymax=df["y"].max(),
        colors="g",
        linestyles="--",
        label="Actual Hit"
    )

    axes[0].vlines(
        bounces,
        ymin=df["y"].min(),
        ymax=df["y"].max(),
        colors="r",
        linestyles="--",
        label="Actual Bounce"
    )

    axes[0].set_title("Actual Hits and Bounces")

    handles, labels = axes[0].get_legend_handles_labels()
    axes[0].legend(dict(zip(labels, handles)).values(),
                dict(zip(labels, handles)).keys())

    # PREDICTED EVENTS
    axes[1].plot(df["frame"], df["y"], lw=1, label="y-position")

    pred_hits = df[df["pred_action"] == "hit"]["frame"]
    pred_bounces = df[df["pred_action"] == "bounce"]["frame"]

    axes[1].vlines(
        pred_hits,
        ymin=df["y"].min(),
        ymax=df["y"].max(),
        colors="g",
        linestyles="--",
        label="Pred Hit"
    )

    axes[1].vlines(
        pred_bounces,
        ymin=df["y"].min(),
        ymax=df["y"].max(),
        colors="r",
        linestyles="--",
        label="Pred Bounce"
    )

    axes[1].set_title("Predicted Hits and Bounces")

    handles, labels = axes[1].get_legend_handles_labels()
    axes[1].legend(dict(zip(labels, handles)).values(),
                dict(zip(labels, handles)).keys())

    plt.xlabel("Frame")
    plt.tight_layout()
    plt.show()

def compute_cm(output_json):
    output = pd.DataFrame(output_json).T
    output = output.reset_index(names=["frame"]).sort_values("frame")
    output["frame"] = output["frame"].astype(int)
    
    hits = output[output.action =="hit"]
    bounces = output[output.action == "bounce"]

    TOL = 5

    num_correct_hit = 0
    num_incorrect_hit = 0
    for i, hit in hits.iterrows():
        frame = hit["frame"]
        
        window  = output[(output.frame >= frame - TOL) & (output.frame <= frame + TOL)]
        match = (window.pred_action == "hit").any()
        
        if match:
            num_correct_hit += 1
        else:
            num_incorrect_hit += 1
            
    num_correct_bounce = 0
    num_incorrect_bounce = 0
    for i, bounce in bounces.iterrows():
        frame = bounce["frame"]
        
        window  = output[(output.frame >= frame - TOL) & (output.frame <= frame + TOL)]
        match = (window.pred_action == "bounce").any()
        
        if match:
            num_correct_bounce += 1
        else:
            num_incorrect_bounce += 1

    cm = pd.DataFrame([[num_correct_hit,num_incorrect_hit],[num_correct_bounce,num_incorrect_bounce]],columns=["Correct","Incorrect"],index=["Hit","Bounce"])
    
    return cm