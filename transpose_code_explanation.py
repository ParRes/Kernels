#!/usr/bin/env python3
"""
Visualization of the transpose operation from PYTHON/transpose.py

This script creates a figure explaining what the code:
    for i in range(order):
        for j in range(order):
            B[i][j] += A[j][i]
            A[j][i] += 1.0

is doing - it shows how matrix elements are being accessed and modified.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_transpose_code_visualization():
    """Create a visualization showing what the transpose code is doing."""
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.25)
    
    # Matrix size for visualization
    n = 5
    
    # Create sample matrices
    A = np.arange(n*n).reshape(n, n).astype(float)
    B = np.zeros((n, n))
    
    # Colors for visualization
    colors = {
        'A_read': '#FFE6E6',      # Light red for A[j][i] (read)
        'A_write': '#FF9999',     # Darker red for A[j][i] (write)
        'B_write': '#E6F3FF',     # Light blue for B[i][j] (write)
        'arrow': '#FF4444',       # Red for arrows
        'grid': '#CCCCCC'         # Gray for grid
    }
    
    # === Matrix A (Source) ===
    ax1.set_title('Matrix A (Source)\nReading A[j][i]', fontsize=14, fontweight='bold')
    
    # Draw matrix A with grid
    im1 = ax1.imshow(A, cmap='Blues', alpha=0.3, origin='upper')
    
    # Add grid
    for i in range(n + 1):
        ax1.axhline(i - 0.5, color=colors['grid'], linewidth=1)
        ax1.axvline(i - 0.5, color=colors['grid'], linewidth=1)
    
    # Add matrix values
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, f'{A[i,j]:.0f}', ha='center', va='center', 
                    fontsize=10, fontweight='bold')
    
    # Highlight transpose access pattern (j,i instead of i,j)
    example_i, example_j = 1, 3
    rect1 = patches.Rectangle((example_i - 0.4, example_j - 0.4), 0.8, 0.8,
                             linewidth=3, edgecolor=colors['arrow'], 
                             facecolor=colors['A_read'], alpha=0.7)
    ax1.add_patch(rect1)
    
    ax1.annotate(f'A[{example_j}][{example_i}]\n(transpose access)', 
                xy=(example_i, example_j), xytext=(example_i + 1.5, example_j - 1.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2),
                fontsize=11, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax1.set_xlim(-0.5, n - 0.5)
    ax1.set_ylim(-0.5, n - 0.5)
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))
    ax1.set_xlabel('j (column index)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('i (row index)', fontsize=12, fontweight='bold')
    ax1.invert_yaxis()
    
    # === Arrow showing the operation ===
    ax2.axis('off')
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    
    # Large arrow
    arrow = patches.FancyArrowPatch((0.1, 0.5), (0.9, 0.5),
                                   arrowstyle='->', mutation_scale=30,
                                   color=colors['arrow'], linewidth=4)
    ax2.add_patch(arrow)
    
    # Code explanation
    code_text = """Code Operation:
    
for i in range(order):
    for j in range(order):
        B[i][j] += A[j][i]
        A[j][i] += 1.0

• Read from A[j][i] (transpose)
• Add to B[i][j] (normal order)
• Increment A[j][i] by 1.0"""
    
    ax2.text(0.5, 0.5, code_text, ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightyellow', alpha=0.8),
            family='monospace')
    
    # === Matrix B (Destination) ===
    ax3.set_title('Matrix B (Destination)\nWriting B[i][j]', fontsize=14, fontweight='bold')
    
    # Simulate the operation result
    B_result = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            B_result[i, j] = A[j, i]  # This is what B[i][j] += A[j][i] does (first iteration)
    
    # Draw matrix B
    im3 = ax3.imshow(B_result, cmap='Oranges', alpha=0.3, origin='upper')
    
    # Add grid
    for i in range(n + 1):
        ax3.axhline(i - 0.5, color=colors['grid'], linewidth=1)
        ax3.axvline(i - 0.5, color=colors['grid'], linewidth=1)
    
    # Add matrix values
    for i in range(n):
        for j in range(n):
            ax3.text(j, i, f'{B_result[i,j]:.0f}', ha='center', va='center', 
                    fontsize=10, fontweight='bold')
    
    # Highlight corresponding write position
    rect3 = patches.Rectangle((example_j - 0.4, example_i - 0.4), 0.8, 0.8,
                             linewidth=3, edgecolor=colors['arrow'], 
                             facecolor=colors['B_write'], alpha=0.7)
    ax3.add_patch(rect3)
    
    ax3.annotate(f'B[{example_i}][{example_j}]\n(normal access)', 
                xy=(example_j, example_i), xytext=(example_j - 1.5, example_i + 1.5),
                arrowprops=dict(arrowstyle='->', color=colors['arrow'], lw=2),
                fontsize=11, fontweight='bold', ha='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    ax3.set_xlim(-0.5, n - 0.5)
    ax3.set_ylim(-0.5, n - 0.5)
    ax3.set_xticks(range(n))
    ax3.set_yticks(range(n))
    ax3.set_xlabel('j (column index)', fontsize=12, fontweight='bold')
    ax3.set_ylabel('i (row index)', fontsize=12, fontweight='bold')
    ax3.invert_yaxis()
    
    # Main title
    fig.suptitle('Matrix Transpose Operation: B[i][j] += A[j][i]; A[j][i] += 1.0',
                fontsize=16, fontweight='bold')
    
    # Add explanation text
    explanation = (
        "The code performs a transpose operation where:\n"
        "• Elements are read from A using transposed indices A[j][i]\n"
        "• These values are accumulated into B using normal indices B[i][j]\n"
        "• Matrix A is modified by incrementing each accessed element by 1.0\n"
        "• This effectively computes B = Aᵀ while updating A"
    )
    
    plt.figtext(0.02, 0.02, explanation, fontsize=11, va='bottom',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

def create_access_pattern_animation():
    """Create an animation showing the access pattern step by step."""
    from matplotlib.animation import FuncAnimation
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.85, bottom=0.15, wspace=0.3)
    
    n = 5  # Matrix size for animation
    A = np.arange(n*n).reshape(n, n).astype(float)
    B = np.zeros((n, n))
    
    def animate_frame(frame):
        ax1.clear()
        ax2.clear()
        
        # Calculate current i, j from frame
        total_ops = n * n
        if frame >= total_ops:
            frame = total_ops - 1
        
        current_i = frame // n
        current_j = frame % n
        
        # Matrix A
        ax1.set_title(f'Matrix A - Reading A[{current_j}][{current_i}]', 
                     fontsize=14, fontweight='bold')
        ax1.imshow(A, cmap='Blues', alpha=0.3, origin='upper')
        
        # Add grid and values
        for i in range(n + 1):
            ax1.axhline(i - 0.5, color='gray', linewidth=1)
            ax1.axvline(i - 0.5, color='gray', linewidth=1)
        
        for i in range(n):
            for j in range(n):
                color = 'red' if (i == current_j and j == current_i) else 'black'
                weight = 'bold' if (i == current_j and j == current_i) else 'normal'
                ax1.text(j, i, f'{A[i,j]:.0f}', ha='center', va='center', 
                        fontsize=12, color=color, fontweight=weight)
        
        # Highlight current read position
        rect1 = patches.Rectangle((current_i - 0.4, current_j - 0.4), 0.8, 0.8,
                                 linewidth=3, edgecolor='red', facecolor='pink', alpha=0.7)
        ax1.add_patch(rect1)
        
        ax1.set_xlim(-0.5, n - 0.5)
        ax1.set_ylim(-0.5, n - 0.5)
        ax1.set_xticks(range(n))
        ax1.set_yticks(range(n))
        ax1.invert_yaxis()
        
        # Matrix B
        ax2.set_title(f'Matrix B - Writing B[{current_i}][{current_j}]', 
                     fontsize=14, fontweight='bold')
        
        # Update B matrix up to current frame
        B_current = np.zeros((n, n))
        for f in range(frame + 1):
            fi = f // n
            fj = f % n
            B_current[fi, fj] += A[fj, fi]
        
        ax2.imshow(B_current, cmap='Oranges', alpha=0.3, origin='upper')
        
        # Add grid and values
        for i in range(n + 1):
            ax2.axhline(i - 0.5, color='gray', linewidth=1)
            ax2.axvline(i - 0.5, color='gray', linewidth=1)
        
        for i in range(n):
            for j in range(n):
                color = 'blue' if (i == current_i and j == current_j) else 'black'
                weight = 'bold' if (i == current_i and j == current_j) else 'normal'
                ax2.text(j, i, f'{B_current[i,j]:.0f}', ha='center', va='center', 
                        fontsize=12, color=color, fontweight=weight)
        
        # Highlight current write position
        rect2 = patches.Rectangle((current_j - 0.4, current_i - 0.4), 0.8, 0.8,
                                 linewidth=3, edgecolor='blue', facecolor='lightblue', alpha=0.7)
        ax2.add_patch(rect2)
        
        ax2.set_xlim(-0.5, n - 0.5)
        ax2.set_ylim(-0.5, n - 0.5)
        ax2.set_xticks(range(n))
        ax2.set_yticks(range(n))
        ax2.invert_yaxis()
        
        # Main title with current operation
        fig.suptitle(f'Step {frame + 1}/{total_ops}: B[{current_i}][{current_j}] += A[{current_j}][{current_i}]  '
                    f'(Value: {A[current_j, current_i]:.0f})',
                    fontsize=16, fontweight='bold')
    
    anim = FuncAnimation(fig, animate_frame, frames=n*n + 5, interval=800, repeat=True)
    return fig, anim

if __name__ == "__main__":
    # Create the static explanation
    print("Creating transpose code explanation...")
    fig1 = create_transpose_code_visualization()
    fig1.savefig('transpose_code_explanation.png', dpi=300, bbox_inches='tight')
    fig1.savefig('transpose_code_explanation.pdf', bbox_inches='tight')
    
    # Create the animated version
    print("Creating access pattern animation...")
    fig2, anim = create_access_pattern_animation()
    anim.save('transpose_access_pattern.gif', writer='pillow', fps=1.2)
    
    # Show the plots
    plt.show()
    
    print("Visualizations saved as:")
    print("- transpose_code_explanation.png (static explanation)")
    print("- transpose_code_explanation.pdf (static explanation)")
    print("- transpose_access_pattern.gif (animated access pattern)")
