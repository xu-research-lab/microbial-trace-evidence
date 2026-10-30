#!/usr/bin/env python3
"""
Visualize SourceTracker2 results for mixed_signal experiment.

Reads mixing_proportions.txt and generates stacked bar charts
showing source contributions to surface_touched samples, 
categorized into 1st subject, 2nd subject, Others, and Unknown.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE = '../'
OUT_DIR = os.path.join(BASE, 'sourcetracker_mixed_signal')
ST2_OUTPUT = os.path.join(OUT_DIR, 'st2_output')

MAP_FP = os.path.join(BASE, 'data/merged_mapping.txt')



DESIRED_ORDER = [
    'H8-H1', 'H1-H8', 
    'H32-H15', 'H15-H32', 
    'H35-H31', 'H31-H35', 
    'H27-H34', 'H34-H27', 
    'H23-H36', 'H36-H23', 
    'H33-H28', 'H28-H33'
]


def main():
    # Load mixing proportions
    mixing_fp = os.path.join(ST2_OUTPUT, 'mixing_proportions.txt')
    if not os.path.exists(mixing_fp):
        print(f"ERROR: {mixing_fp} not found. Run SourceTracker 1st subject.")
        return

    mp = pd.read_table(mixing_fp, index_col=0)
    print(f"Mixing proportions shape: {mp.shape}")
    print(f"Columns (sources): {list(mp.columns)}")
    print(f"Rows (sinks): {list(mp.index)}")

    # Load mapping for metadata
    md = pd.read_table(MAP_FP, index_col=0,
                       dtype={'month': str, 'day': str, 'year': str, 'sample_date': str})

    # Identify sink samples and filter out controls with 'unknown' pair
    sink_ids = [s for s in mp.index if s in md.index and md.loc[s, 'sample_type'] == 'surface_touched']
    sink_ids = [s for s in sink_ids
                if str(md.loc[s, 'pair1_resident']).strip().lower() != 'unknown'
                and str(md.loc[s, 'pair2_perp']).strip().lower() != 'unknown']

    # Sort based on the custom DESIRED_ORDER list
    def custom_sort_key(s):
        p1 = str(md.loc[s, 'pair1_resident']) if pd.notna(md.loc[s, 'pair1_resident']) else ''
        p2 = str(md.loc[s, 'pair2_perp']) if pd.notna(md.loc[s, 'pair2_perp']) else ''
        label = f"{p1}-{p2}"
        
        # If the label is in our list, use its index. Otherwise, push it to the end (999)
        order_idx = DESIRED_ORDER.index(label) if label in DESIRED_ORDER else 999
        stype = str(md.loc[s, 'surface_type'])
        return (stype, order_idx, s)

    sink_ids.sort(key=custom_sort_key)
    sink_mp = mp.loc[sink_ids]

    # Identify known sources (individuals that actually appear in sink pairs)
    all_candidates = [c for c in mp.columns if c.startswith('H')]
    used_individuals = set()
    for s in sink_ids:
        for col in ['pair1_resident', 'pair2_perp']:
            val = str(md.loc[s, col])
            if val.strip().lower() not in ('unknown', 'nan', ''):
                used_individuals.add(val)
    individual_sources = [c for c in all_candidates if c in used_individuals]
    removed_sources = [c for c in all_candidates if c not in used_individuals]
    print(f"\nIndividual sources (kept): {sorted(individual_sources)}")
    print(f"Source individuals removed (not in any sink pair): {sorted(removed_sources)}")


    # === Plot 1: Stacked bar categorized by source type (Faceted) ===
    print("\nGenerating faceted stacked bar chart...")
    
    # Define colors for the four categories
    category_colors = {
    '1st subject': '#D62728',    
    '2nd subject': '#1F77B4',    
    'Others':      '#A69B8F',    
    'Unknown':     '#D3D3D3'     
    }

    # Determine the exact name of the 'Unknown' column in the SourceTracker output
    unk_col = 'Unknown' if 'Unknown' in mp.columns else ('unknown' if 'unknown' in mp.columns else None)

    surf_types = ['ceramic', 'plastic']
    
    # Pre-calculate sample IDs for each surface to adjust facet widths dynamically
    surf_samples = {}
    for st in surf_types:
        surf_samples[st] = [s for s in sink_mp.index if s in md.index and md.loc[s, 'surface_type'] == st]
    
    # Calculate width ratios (avoid zero width if a category is empty)
    widths = [max(1, len(surf_samples[st])) for st in surf_types]
    
    # Create a single figure with 2 subplots (facets), sharing the Y-axis
    fig1, axes = plt.subplots(1, 2, figsize=(5.5, 2), 
                              sharey=True, gridspec_kw={'width_ratios': widths,'wspace': 0.05})

    for ax, surf_type in zip(axes, surf_types):
        surf_ids = surf_samples[surf_type]
        
        if not surf_ids:
            ax.text(0.5, 0.5, f'No samples', ha='center', va='center')
            ax.set_title(f'{surf_type.capitalize()}')
            ax.set_xticks([])
            continue

        # Prepare data containers
        cat_data = {'1st subject': [], '2nd subject': [], 'Others': [], 'Unknown': []}
        labels_surf = []

        # Calculate proportions for each category sample by sample
        for s in surf_ids:
            p1 = str(md.loc[s, 'pair1_resident']) if pd.notna(md.loc[s, 'pair1_resident']) else ''
            p2 = str(md.loc[s, 'pair2_perp']) if pd.notna(md.loc[s, 'pair2_perp']) else ''
            labels_surf.append(f"{p1}-{p2}")

            val_unk = mp.loc[s, unk_col] if unk_col else 0.0
            val_1st_subject = mp.loc[s, p1] if p1 in mp.columns else 0.0
            val_2nd_subject = mp.loc[s, p2] if p2 in mp.columns else 0.0

            # 'Others' represents the sum of all remaining sources
            val_others = 0.0
            for col in mp.columns:
                if col not in [p1, p2, unk_col]:
                    val_others += mp.loc[s, col]

            cat_data['1st subject'].append(val_1st_subject)
            cat_data['2nd subject'].append(val_2nd_subject)
            cat_data['Others'].append(val_others)
            cat_data['Unknown'].append(val_unk)

        bottom = np.zeros(len(surf_ids))
        plot_order = ['1st subject', '2nd subject', 'Others', 'Unknown']
        
        for cat in plot_order:
            vals = np.array(cat_data[cat])
            # Only add label to the 1st subject axis to avoid duplicate legend entries
            ax.bar(range(len(surf_ids)), vals, bottom=bottom, 
                   label=cat if ax == axes[0] else "", 
                   color=category_colors[cat])
            bottom += vals

        ax.set_xticks(range(len(surf_ids)))
        ax.set_xticklabels(
            labels_surf, 
            rotation=45, 
            ha='right',           
            rotation_mode='anchor',fontsize= 9 
        )
        ax.set_title(f'{surf_type.capitalize()}')
        ax.set_ylim(0, 1)
        
        # Only set Y label on the leftmost facet
        if ax == axes[0]:
            ax.set_ylabel('Proportion')

    # Add a single main title and place the legend outside the overall plot
    handles, labels = axes[0].get_legend_handles_labels()
    num_labels = len(labels)
    fig1.legend(
        handles, 
        labels, 
        loc='lower center',          
        bbox_to_anchor=(0.5, -0.4),   
        ncol=num_labels,             
        fontsize=9,             
        handlelength=1.2,            
        handletextpad=0.5,           
        columnspacing=1.5           
    )
    plt.tight_layout()

    # Save the faceted figure
    fname_stacked = 'sourcetracker_stacked_bar_faceted'
    fig1.savefig(os.path.join(OUT_DIR, f'{fname_stacked}.pdf'), bbox_inches='tight')
    fig1.savefig(os.path.join(OUT_DIR, f'{fname_stacked}.png'), dpi=300, bbox_inches='tight')
    print(f"Saved: {fname_stacked}.pdf/png")



    plt.close('all')
    print("\nDone!")


if __name__ == '__main__':
    main()