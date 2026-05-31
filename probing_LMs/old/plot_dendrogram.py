import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as path_effects

def create_custom_dendrogram_heatmap(
    data_df, 
    data_df_full,
    row_hierarchy, 
    col_hierarchy, 
    figsize=(30, 20), 
    dpi=150
):
    """
    Creates a heatmap with custom (multi-branch) dendrograms on the left (rows)
    and bottom (columns). Ensures:
      - The row dendrogram exactly matches the heatmap's vertical extent.
      - The column dendrogram exactly matches the heatmap's horizontal extent.
      - No unwanted capping of the color scale (it will use the actual min/max).
      - More spacing for the column dendrogram levels; labels not tilted.
      - Bigger, bolder top-level labels; colored label boxes based on hierarchy level.
    """
    # ------------------------------------------------------------
    # 1) Prepare figure and axes; we'll initially let them overlap
    #    then manually position them to ensure perfect alignment.
    # ------------------------------------------------------------
    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Create main axes for the heatmap (we'll reposition later if needed).
    ax_heatmap    = fig.add_axes([0.2, 0.2, 0.6, 0.6])  
    ax_row_dendro = fig.add_axes([0.1, 0.2, 0.1, 0.6])  
    ax_col_dendro = fig.add_axes([0.2, 0.1, 0.6, 0.1])  
    ax_cbar       = fig.add_axes([0.81, 0.2, 0.02, 0.6]) # colorbar on the right

    # ------------------------------------------------------------
    # 2) Build label->position mappings (row 0 at top, col 0 at left).
    #    We'll offset brackets by +0.5 to center on each heatmap cell.
    # ------------------------------------------------------------
    row_label_to_pos = {lab: i for i, lab in enumerate(data_df.index)}
    col_label_to_pos = {lab: i for i, lab in enumerate(data_df.columns)}

    # We'll store (start, end, depth, label)
    # plus text positions: (mid, depth, label).
    def labeled_traverse(node, parent_label, label_to_pos_map, depth, connections, level_texts):
        """
        Recursively traverse the user-defined hierarchy,
        building bracket intervals + text for each group.
        """
        positions = []
        if isinstance(node, dict):
            # Dict => subgroups
            for sublabel, subnode in node.items():
                child_positions = labeled_traverse(subnode, sublabel, label_to_pos_map,
                                                   depth+1, connections, level_texts)
                positions.extend(child_positions)
            if positions:
                mn, mx = min(positions), max(positions)
                connections.append((mn, mx, depth, parent_label))
                level_texts.append(((mn + mx)/2.0, depth, parent_label))
            return positions

        elif isinstance(node, list):
            # List => leaves
            leaf_positions = []
            for leaf_label in node:
                if leaf_label in label_to_pos_map:
                    leaf_positions.append(label_to_pos_map[leaf_label])
            if leaf_positions:
                mn, mx = min(leaf_positions), max(leaf_positions)
                connections.append((mn, mx, depth, parent_label))
                level_texts.append(((mn + mx)/2.0, depth, parent_label))
            return leaf_positions

        return positions

    # Row hierarchy
    row_connections = []
    row_texts = []
    for top_label, subtree in row_hierarchy.items():
        labeled_traverse(subtree, top_label, row_label_to_pos, 0,
                         row_connections, row_texts)
    max_row_level = max((c[2] for c in row_connections), default=0)

    # Column hierarchy
    col_connections = []
    col_texts = []
    for top_label, subtree in col_hierarchy.items():
        labeled_traverse(subtree, top_label, col_label_to_pos, 0,
                         col_connections, col_texts)
    max_col_level = max((c[2] for c in col_connections), default=0)

    # ------------------------------------------------------------
    # 3) Plot the heatmap
    #    - We explicitly set vmin/vmax to the true min/max to avoid capping.
    # ------------------------------------------------------------
    sns.heatmap(
        data_df,
        ax=ax_heatmap,
        annot=True,
        fmt=".2f",
        cmap='Reds',
        vmin=data_df_full.values.min(),
        vmax=data_df_full.values.max(),
        square=True,
        linewidths=0.1,
        annot_kws={"size": 16},
        cbar_ax=ax_cbar,
        cbar_kws={"shrink": 1.0, "extend": "both"}  # helps ensure full range
    )
    ax_heatmap.set_xticklabels(
        ax_heatmap.get_xticklabels(), 
        fontsize=28, 
        rotation=90, 
        ha="center"
    )
    ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), fontsize=28)
    ax_heatmap.set_xlabel("")

    # ------------------------------------------------------------
    # 4) Row dendrogram (ax_row_dendro)
    #    - Perfectly match height to the heatmap's y-range
    #    - Largest depth near the heatmap, top-level is further left
    # ------------------------------------------------------------
    ax_row_dendro.set_ylim(*ax_heatmap.get_ylim()) 
    # If the heatmap is oriented with row0 at the top, we invert the row axis:
    # ax_row_dendro.invert_yaxis()
    ax_row_dendro.axis('off')

    # We'll define a function to get the color for each label box
    # based on the "depth." Feel free to change these colors:
    level_colors = [
        "lightgrey", "white", "lightgrey"
    ]
    def get_level_color(d):
        """Return a color (string) for a given dendrogram depth."""
        if d < len(level_colors):
            return level_colors[d]
        return level_colors[-1]

    # We'll space the brackets in X:
    # e.g. x_right = 0.9, x_left = 0.9 - factor*(max_row_level-depth)
    bracket_space = 0.7  # total horizontal space for the row dendrogram
    for (start_idx, end_idx, depth, label) in row_connections:
        # Move bracket to center of cells => +0.5
        y1 = start_idx + 0.5
        y2 = end_idx   + 0.5
        if y1 > y2:
            y1, y2 = y2, y1
        inv = max_row_level - depth
        x_right = 0.95
        x_left  = x_right - (inv+1)*(bracket_space/(max_row_level+1))

        ax_row_dendro.plot([x_left, x_right], [y1, y1], color='black', lw=3.5)
        ax_row_dendro.plot([x_left, x_right], [y2, y2], color='black', lw=3.5)
        ax_row_dendro.plot([x_left, x_left], [y1, y2], color='black', lw=3.5)

    # Label each bracket
    for (mid_idx, depth, label) in row_texts:
        y_mid = mid_idx + 0.5
        inv = max_row_level - depth
        x_text = 0.95 - (inv+1)*(bracket_space/(max_row_level+1)) - 0.01
        # Slightly bigger text for top-level
        if depth == 0:
            fsize = 26
            fw = 'bold'
        else:
            fsize = 21
            fw = 'bold'

        color_box = get_level_color(depth)
        bbox_props = dict(
            boxstyle="round,pad=0.3", 
            fc=color_box,   # color by depth
            ec="black", 
            lw=3, 
            alpha=1.0       # opaque
        )
        txt = ax_row_dendro.text(
            x_text,
            y_mid,
            label,
            ha='center',
            va='center',
            fontsize=fsize,
            fontweight=fw,
            bbox=bbox_props
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    # ------------------------------------------------------------
    # 5) Column dendrogram (ax_col_dendro)
    #    - Perfectly match width to the heatmap's x-range
    #    - Largest depth near the heatmap, top-level further down
    # ------------------------------------------------------------
    ax_col_dendro.set_xlim(*ax_heatmap.get_xlim())
    ax_col_dendro.axis('off')

    # We'll allow more vertical space for deeper levels:
    col_bracket_space = .8  # total vertical space
    for (start_idx, end_idx, depth, label) in col_connections:
        x1 = start_idx + 0.5
        x2 = end_idx   + 0.5
        if x1 > x2:
            x1, x2 = x2, x1
        inv = max_col_level - depth
        y_top = 0.0
        y_bottom = y_top - (inv+1)*(col_bracket_space/(max_col_level+1))

        ax_col_dendro.plot([x1, x1], [y_bottom, y_top], color='black', lw=3.5)
        ax_col_dendro.plot([x2, x2], [y_bottom, y_top], color='black', lw=3.5)
        ax_col_dendro.plot([x1, x2], [y_bottom, y_bottom], color='black', lw=3.5)

    for (mid_idx, depth, label) in col_texts:
        x_mid = mid_idx + 0.5
        inv = max_col_level - depth
        y_text = -(inv+1)*(col_bracket_space/(max_col_level+1)) - 0.01
        if depth == 0:
            fsize = 26
            fw = 'bold'
        else:
            fsize = 21
            fw = 'bold'

        color_box = get_level_color(depth)
        bbox_props = dict(
            boxstyle="round,pad=0.3", 
            fc=color_box,   # colored by depth
            ec="black", 
            lw=3, 
            alpha=1.0       # opaque
        )
        txt = ax_col_dendro.text(
            x_mid,
            y_text,
            label,
            ha='center',
            va='center',
            rotation=0,            # not tilted
            fontsize=fsize,
            fontweight=fw,
            bbox=bbox_props
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    # ------------------------------------------------------------
    # 6) Final layout adjustments:
    #    - We'll let the colorbar stand as is; you can tweak x/w if desired.
    # ------------------------------------------------------------
    # If you want the dendrogram axes to *exactly* match the heatmap bounding box,
    # you can do final bounding-box alignment here:
    fig.canvas.draw()  # ensure correct positions after drawing
    heatmap_pos = ax_heatmap.get_position()

    # Match row dendrogram to the same top/bottom as heatmap:
    # (width can be adjusted to your preference)
    ax_row_dendro.set_position([
        heatmap_pos.x0 - 0.185,  # shift left
        heatmap_pos.y0,        
        0.08,                   # dendrogram width
        heatmap_pos.height      # same height
    ])

    # Match col dendrogram to the same left/right as heatmap:
    ax_col_dendro.set_position([
        heatmap_pos.x0,
        heatmap_pos.y0 - 0.22,  # shift down
        heatmap_pos.width,
        0.10                    # dendrogram height
    ])

    # Move colorbar to the right of the heatmap:
    ax_cbar.set_position([
        heatmap_pos.x0 + heatmap_pos.width + 0.01,
        heatmap_pos.y0,
        0.02,
        heatmap_pos.height
    ])

    return fig

# -----------------------------------------------------------------
# Example usage (assuming you already have the DataFrame 'average_masked'):
row_hierarchy = {
    "Behavior": {
        "Eye-tracking": ["Provo", "Meco-en", "ZuCo", "UCL$_{ET}$"],
        "Self-paced R.": ["UCL$_{SPR}$", "Brown", "NatStor, SPR"],
        "Maze": ["NatStor, maze"]
    },
    "Brain": {
        "EEG": ["UCL$_{N400}$"],
        "fMRI": ["Harry Potter", "NatStor, fMRI", "Pereira", "Control"]
    }
}
col_hierarchy = {
    "Form": {
    "Surface": ["SentLen", "Surprisal", "Frequency", "Length"],
    "Syntax": ["BShift", "TDepth"]},
    "Semantics": {
        "Sensory": [
            "Auditory", "Visual", "Gustatory", "Olfactory",
            "Haptic", "Enteroception", "Max$_{perc.}$"
        ],
        "Motor": [
            "Foot", "Hand", "Head", "Mouth", "Torso", "Max$_{act.}$"
        ],
        "Affective": [
            "Valence", "Arousal", "Dominance", "Social"
        ],
        "Sentence": [
            "SOMO", "ObjN", "SubjN", "Tense"
        ]
    },
    "Discourse": ["CoordInv"]
}
fig = create_custom_dendrogram_heatmap(average_masked, average, row_hierarchy, col_hierarchy)
plt.savefig('custom_dendrogram_heatmap.png', bbox_inches='tight', dpi=300)
plt.show()
