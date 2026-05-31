import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.patheffects as path_effects

def create_custom_dendrogram_heatmap(
    data_df, 
    data_df_full,  # for the true min/max of color scale
    row_hierarchy, 
    col_hierarchy, 
    figsize=(30, 20), 
    dpi=150
):
    """
    Creates a heatmap with custom (multi-branch) dendrograms on the left (rows)
    and bottom (columns). Key points:
      - We reintroduce numeric annotations on the heatmap cells.
      - The final (leaf-level) labels on the Y axis are right-aligned.
      - The final (leaf-level) labels on the X axis are vertical.
      - Higher-level group labels remain horizontally oriented and centered.
      - The top-level items in `row_hierarchy` appear at the top in the order
        that `data_df` is indexed (no Y-axis inversion).
      - No standard tick labels on the heatmap; leaf labels are drawn by the dendrogram.
    """

    fig = plt.figure(figsize=figsize, dpi=dpi)

    # Create main axes for the heatmap (we'll reposition later).
    ax_heatmap    = fig.add_axes([0.2, 0.2, 0.6, 0.6])
    ax_row_dendro = fig.add_axes([0.1, 0.2, 0.1, 0.6])
    ax_col_dendro = fig.add_axes([0.2, 0.1, 0.6, 0.1])
    ax_cbar       = fig.add_axes([0.81, 0.2, 0.02, 0.6])

    # Build maps from row/col labels to integer positions.
    row_label_to_pos = {lab: i for i, lab in enumerate(data_df.index)}
    col_label_to_pos = {lab: i for i, lab in enumerate(data_df.columns)}

    # We'll store bracket intervals as (start, end, depth, label) and
    # also store text positions as (mid, depth, label).
    def labeled_traverse(node, parent_label, label_to_pos_map, depth, connections, level_texts):
        """
        Recursively traverse a nested hierarchy (dict or list),
        building bracket intervals + text for each group.
        We also add per-leaf brackets at depth+1 so each leaf
        can have its own label drawn as well.
        """
        positions = []
        if isinstance(node, dict):
            # Subgroups
            for sublabel, subnode in node.items():
                child_positions = labeled_traverse(
                    subnode,
                    sublabel,
                    label_to_pos_map,
                    depth+1,
                    connections,
                    level_texts
                )
                positions.extend(child_positions)

            if positions:
                mn, mx = min(positions), max(positions)
                connections.append((mn, mx, depth, parent_label))
                level_texts.append(((mn + mx)*0.5, depth, parent_label))

            return positions

        elif isinstance(node, list):
            # Leaves
            leaf_positions = []
            for leaf_label in node:
                if leaf_label in label_to_pos_map:
                    leaf_positions.append(label_to_pos_map[leaf_label])

            if leaf_positions:
                mn, mx = min(leaf_positions), max(leaf_positions)
                # one bracket for the entire list
                connections.append((mn, mx, depth, parent_label))
                level_texts.append(((mn + mx)*0.5, depth, parent_label))

                # each leaf gets its own bracket at depth+1
                for lf_label in node:
                    if lf_label in label_to_pos_map:
                        lf_pos = label_to_pos_map[lf_label]
                        connections.append((lf_pos, lf_pos, depth+1, lf_label))
                        level_texts.append((lf_pos, depth+1, lf_label))

            return leaf_positions

        return positions

    # Build row hierarchy
    row_connections = []
    row_texts = []
    for top_label, subtree in row_hierarchy.items():
        labeled_traverse(subtree, top_label, row_label_to_pos, 0, row_connections, row_texts)
    max_row_level = max((c[2] for c in row_connections), default=0)

    # Build column hierarchy
    col_connections = []
    col_texts = []
    for top_label, subtree in col_hierarchy.items():
        labeled_traverse(subtree, top_label, col_label_to_pos, 0, col_connections, col_texts)
    max_col_level = max((c[2] for c in col_connections), default=0)

    # ----------------------------------------------------------------
    # Identify which connections are leaf-level (start==end) for row/col
    # so we can handle label alignment, rotation, etc.
    # We'll store them in sets for quick membership checks.
    # ----------------------------------------------------------------
    row_leaf_connections = set()
    for (st, en, dp, lab) in row_connections:
        if st == en:  # leaf
            row_leaf_connections.add((st, dp, lab))

    col_leaf_connections = set()
    for (st, en, dp, lab) in col_connections:
        if st == en:
            col_leaf_connections.add((st, dp, lab))

    # 5) Plot heatmap with numeric annotations
    sns.heatmap(
        data_df,
        ax=ax_heatmap,
        #annot=True,
        #fmt=".2f",
        #annot_kws={'size': 15},
        cmap='Reds',
        vmin=data_df_full.values.min(),
        vmax=data_df_full.values.max(),
        square=True,
        linewidths=4,
        cbar_ax=ax_cbar,
        cbar_kws={"shrink": 1.0, "extend": "both"}
    )
    # Remove standard tick labels (we rely on dendrogram leaf labels).
    ax_heatmap.set_xticks([])
    ax_heatmap.set_yticks([])

    # ----------------------------------------------------------------
    # Row dendrogram
    #   - The top row in data_df appears on top, so no invert_yaxis.
    # ----------------------------------------------------------------
    ax_row_dendro.set_ylim(*ax_heatmap.get_ylim())
    ax_row_dendro.axis('off')

    # Colors by depth
    level_colors = ["lightgrey", "white", "white"]
    def get_level_color(d):
        if d < len(level_colors):
            return level_colors[d]
        return level_colors[-1]

    # Draw row brackets
    bracket_space = 700
    for (start_idx, end_idx, depth, label) in row_connections:
        if depth == 2:
            continue
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

    # Draw row labels
    for (mid_idx, depth, label) in row_texts:
        y_mid = mid_idx + 0.5
        inv   = max_row_level - depth
        x_text = 0.95 - (inv+1)*(bracket_space/(max_row_level+1)) - 0.01

        # detect if this is a leaf-level label
        # We'll check if there's a connection with start==end at that index, depth, label
        # Because mid_idx might be floating, we do round(mid_idx) to match the integer bracket pos.
        is_leaf = (round(mid_idx), depth, label) in row_leaf_connections

        if depth == 0:
            # top-level
            fsize = 26
            fw    = 'bold'
        else:
            # subgroup or leaf
            fsize = 21
            fw    = 'bold'

        color_box = get_level_color(depth)
        bbox_props = dict(
            boxstyle="round,pad=0.3",
            fc=color_box,
            ec="black",
            lw=3,
            alpha=1.0
        )
        # If leaf => align to right, else center
        ha_style = 'right' if depth == 2 else 'center'
        add_x = 230 if depth == 2 else 0
        txt = ax_row_dendro.text(
            x_text+add_x, 
            y_mid,
            label,
            ha=ha_style,
            va='center',
            fontsize=fsize,
            fontweight=fw,
            bbox=bbox_props
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    # ----------------------------------------------------------------
    # Column dendrogram
    #   - largest depth near the heatmap, top-level further down
    # ----------------------------------------------------------------
    ax_col_dendro.set_xlim(*ax_heatmap.get_xlim())
    ax_col_dendro.axis('off')

    col_bracket_space = 700
    for (start_idx, end_idx, depth, label) in col_connections:
        if depth == 2:
            continue
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
        

        # detect leaf-level
        is_leaf = depth >= 2 #(round(mid_idx), depth, label) in col_leaf_connections
        if label == "CoordInv":
            is_leaf = True
            depth = 2
            
        x_mid = mid_idx + 0.5
        inv   = max_col_level - depth
        y_text = -(inv+1)*(col_bracket_space/(max_col_level+1)) - 0.01
        
        if label == "Discourse":
            x_mid = x_mid + 1.05

        if depth == 0:
            fsize = 26
            fw    = 'bold'
        else:
            fsize = 21
            fw    = 'bold'

        color_box = get_level_color(depth)
        bbox_props = dict(
            boxstyle="round,pad=0.3",
            fc=color_box,
            ec="black",
            lw=3,
            alpha=1.0
        )
        # If leaf => rotate vertical, else keep horizontal
        
        if is_leaf:
            rotation_angle = 90
            ha_style = 'center'
            va_style = 'top'
        else:
            rotation_angle = 0
            ha_style = 'center'
            va_style = 'center'
        add_y = 220 if is_leaf else 0
        txt = ax_col_dendro.text(
            x_mid,
            y_text+add_y,
            label,
            ha=ha_style,
            va=va_style,
            rotation=rotation_angle,
            fontsize=fsize,
            fontweight=fw,
            bbox=bbox_props
        )
        txt.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    # ----------------------------------------------------------------
    # Final layout: match dendrogram axes to the heatmap bounding box
    # ----------------------------------------------------------------
    fig.canvas.draw()
    heatmap_pos = ax_heatmap.get_position()

    # row dendro: same top/bottom as heatmap, shift left
    # Make the row dendrogram axis wider and shift it left
    ax_row_dendro.set_position([
        heatmap_pos.x0 - 0.2,   # shift more to the left
        heatmap_pos.y0,
        0.2,                    # increase width from 0.08 to 0.2
        heatmap_pos.height
    ])


    # col dendro: same left/right as heatmap, shift down
    ax_col_dendro.set_position([
    heatmap_pos.x0,
    heatmap_pos.y0 - 0.25,   # shift further down
    heatmap_pos.width,
    0.25                     # increase the height from 0.1
    ])


    # colorbar on the right
    ax_cbar.set_position([
        heatmap_pos.x0 + heatmap_pos.width + 0.01,
        heatmap_pos.y0,
        0.02,
        heatmap_pos.height
    ])

    return fig

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

# Example usage (assuming you already have 'average_masked' and 'average'):
fig = create_custom_dendrogram_heatmap(
    average_masked,
    average,
    row_hierarchy,
    col_hierarchy
)
plt.savefig('custom_dendrogram_heatmap.png', bbox_inches='tight', dpi=300)
plt.show()
