#!/usr/bin/env python3
"""
Visualization of Matrix Transpose Layout from PRK MPI1/Transpose/transpose-a2a.c

This script creates a professional-looking diagram showing the matrix layout
for distributed transpose operations, corresponding to lines 95-119 of the
transpose-a2a.c file.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_transpose_layout_diagram():
    """Create a diagram showing the matrix layout for distributed transpose."""
    
    # Set up the figure with a clean, professional style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Define the overall matrix dimensions and layout
    matrix_width = 12
    matrix_height = 8
    num_ranks = 8  # Number of ranks (column blocks)
    num_blocks_per_rank = 8  # Number of blocks per column block
    
    # Calculate dimensions
    colblock_width = matrix_width / num_ranks
    block_height = matrix_height / num_blocks_per_rank
    
    # Colors for different elements
    colors = {
        'overall_matrix': '#E8F4FD',
        'colblock': '#B3D9FF',
        'block': '#FF9999',
        'block_border': '#CC0000',
        'colblock_border': '#0066CC',
        'overall_border': '#000000'
    }
    
    # Draw the overall matrix background
    overall_rect = patches.Rectangle(
        (0, 0), matrix_width, matrix_height,
        linewidth=3, edgecolor=colors['overall_border'],
        facecolor=colors['overall_matrix'], alpha=0.3
    )
    ax.add_patch(overall_rect)
    
    # Draw column blocks (Colblocks)
    for rank in range(num_ranks):
        x_start = rank * colblock_width
        
        # Draw colblock background
        colblock_rect = patches.Rectangle(
            (x_start, 0), colblock_width, matrix_height,
            linewidth=2, edgecolor=colors['colblock_border'],
            facecolor=colors['colblock'], alpha=0.5,
            linestyle='--'
        )
        ax.add_patch(colblock_rect)
        
        # Highlight one specific colblock (rank 2)
        if rank == 2:
            # Draw blocks within this colblock
            for block in range(num_blocks_per_rank):
                y_start = block * block_height
                
                block_rect = patches.Rectangle(
                    (x_start, y_start), colblock_width, block_height,
                    linewidth=2.5, edgecolor=colors['block_border'],
                    facecolor=colors['block'], alpha=0.7
                )
                ax.add_patch(block_rect)
                
                # Add block label
                ax.text(x_start + colblock_width/2, y_start + block_height/2,
                       f'Block {block}', ha='center', va='center',
                       fontsize=10, fontweight='bold', color='white')
            
            # Add colblock label
            ax.text(x_start + colblock_width/2, matrix_height + 0.3,
                   'Colblock\n(One Rank)', ha='center', va='bottom',
                   fontsize=12, fontweight='bold', color=colors['colblock_border'])
        else:
            # Draw block divisions for other colblocks (lighter)
            for block in range(num_blocks_per_rank):
                y_start = block * block_height
                
                block_rect = patches.Rectangle(
                    (x_start, y_start), colblock_width, block_height,
                    linewidth=1, edgecolor=colors['block_border'],
                    facecolor='none', alpha=0.3, linestyle=':'
                )
                ax.add_patch(block_rect)
    
    # Add labels and annotations
    ax.text(matrix_width/2, -0.8, 'Overall Matrix', ha='center', va='top',
            fontsize=14, fontweight='bold', color=colors['overall_border'])
    
    # Add rank labels
    for rank in range(num_ranks):
        x_center = rank * colblock_width + colblock_width/2
        ax.text(x_center, -0.3, f'Rank {rank}', ha='center', va='top',
                fontsize=10, color=colors['colblock_border'])
    
    # Add arrows and explanatory text
    arrow_props = dict(arrowstyle='->', connectionstyle='arc3,rad=0.1',
                      color='red', lw=2)
    
    # Arrow pointing to the highlighted colblock
    ax.annotate('Each rank owns one\ncolumn block (Colblock)',
                xy=(3.0, matrix_height/2), xytext=(9, 6),
                arrowprops=arrow_props, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8))
    
    # Arrow pointing to a block
    ax.annotate('Blocks are units of\ncommunication between ranks',
                xy=(3.0, block_height/2), xytext=(9, 2),
                arrowprops=arrow_props, fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.8))
    
    # Set up the plot
    ax.set_xlim(-1, matrix_width + 4)
    ax.set_ylim(-1.5, matrix_height + 1)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Add title
    plt.title('Matrix Layout for Distributed Transpose Operation\n'
              'All-to-All Communication Pattern',
              fontsize=16, fontweight='bold', pad=20)
    
    # Add description text
    description = (
        "• Each rank owns one Colblock (column block) stored contiguously in memory\n"
        "• Colblocks are subdivided into Blocks for communication\n"
        "• Block i of rank j is sent to rank i during transpose operation\n"
        "• Storage format: column major (elements (i,j) and (i+1,j) are adjacent)"
    )
    
    plt.figtext(0.02, 0.02, description, fontsize=10, va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_transpose_animation():
    """Create an animated version showing the transpose process."""
    from matplotlib.animation import FuncAnimation
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3)
    
    # Matrix dimensions
    n_ranks = 8
    matrix_size = 8
    block_size = matrix_size // n_ranks
    
    def animate_frame(frame):
        ax1.clear()
        ax2.clear()
        
        # Original matrix (left)
        ax1.set_title('Original Matrix A', fontsize=14, fontweight='bold')
        original_matrix = np.arange(matrix_size * matrix_size).reshape(matrix_size, matrix_size)
        
        # Show the matrix with rank ownership using consistent colors
        base_colors = ['Blues', 'Reds', 'Greens', 'Oranges', 
                      'Purples', 'Greys', 'YlOrBr', 'BuGn']
        for rank in range(n_ranks):
            start_col = rank * block_size
            end_col = (rank + 1) * block_size
            
            # Highlight current rank's colblock
            alpha = 0.8 if frame % n_ranks == rank else 0.3
            ax1.imshow(original_matrix[:, start_col:end_col], 
                      extent=[start_col, end_col, 0, matrix_size],
                      cmap=base_colors[rank % len(base_colors)],
                      alpha=alpha, origin='upper')
        
        # Draw grid
        for i in range(matrix_size + 1):
            ax1.axhline(i, color='black', linewidth=0.5)
            ax1.axvline(i, color='black', linewidth=0.5)
        
        # Highlight rank boundaries
        for rank in range(1, n_ranks):
            ax1.axvline(rank * block_size, color='red', linewidth=3)
        
        ax1.set_xlim(0, matrix_size)
        ax1.set_ylim(0, matrix_size)
        ax1.invert_yaxis()  # Make (0,0) at top-left
        ax1.set_aspect('equal')
        
        # Add x-axis labels at the top
        ax1.xaxis.tick_top()
        ax1.xaxis.set_label_position('top')
        ax1.set_xlabel('Column Index', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Row Index', fontsize=12, fontweight='bold')
        
        # Transposed matrix (right) - now organized by rows
        ax2.set_title('Transposed Matrix B', fontsize=14, fontweight='bold')
        transposed_matrix = original_matrix.T
        
        for rank in range(n_ranks):
            start_row = rank * block_size
            end_row = (rank + 1) * block_size
            
            # Show data movement - use same colors as matrix A for consistency
            # But we need to map the colors correctly for the transposed data
            alpha = 0.8 if frame % n_ranks == rank else 0.3
            
            # Get the corresponding column from the original matrix to maintain color consistency
            original_col_data = original_matrix[:, rank * block_size:(rank + 1) * block_size]
            transposed_row_data = original_col_data.T  # This gives us the correct orientation
            
            colormap_name = base_colors[rank % len(base_colors)]
            reversed_colormap = colormap_name + '_r'  # Add '_r' suffix to reverse the colormap
            
            ax2.imshow(transposed_row_data,
                      extent=[0, matrix_size, start_row, end_row],
                      cmap=reversed_colormap,
                      alpha=alpha, origin='upper')
        
        # Draw grid
        for i in range(matrix_size + 1):
            ax2.axhline(i, color='black', linewidth=0.5)
            ax2.axvline(i, color='black', linewidth=0.5)
        
        for rank in range(1, n_ranks):
            ax2.axhline(rank * block_size, color='red', linewidth=3)
        
        ax2.set_xlim(0, matrix_size)
        ax2.set_ylim(0, matrix_size)
        ax2.invert_yaxis()  # Make (0,0) at top-left
        ax2.set_aspect('equal')
        
        # Add x-axis labels at the top
        ax2.xaxis.tick_top()
        ax2.xaxis.set_label_position('top')
        ax2.set_xlabel('Column Index', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Row Index', fontsize=12, fontweight='bold')
        
        # Add frame information
        current_rank = frame % n_ranks
        fig.suptitle(f'Matrix Transpose Communication - Processing Rank {current_rank}',
                    fontsize=16, fontweight='bold')
    
    anim = FuncAnimation(fig, animate_frame, frames=n_ranks*3, interval=1000, repeat=True)
    return fig, anim

if __name__ == "__main__":
    # Create the animated version only
    print("Creating transpose animation...")
    fig, anim = create_transpose_animation()
    anim.save('transpose_animation.gif', writer='pillow', fps=1)
    
    # Show the plot
    plt.show()
    
    print("Animation saved as:")
    print("- transpose_animation.gif (animated transpose process)")
