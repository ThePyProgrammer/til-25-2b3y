import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle

def visualize_map(map_obj, show_agents=True, show_points=True, show_visited=True, figsize=(10, 10)):
    """
    Visualize the map reconstructed from observations.
    
    Args:
        map_obj: The Map object containing the reconstructed map
        show_agents: Whether to show agent positions
        show_points: Whether to show recon and mission points
        show_visited: Whether to show visited locations
        figsize: Size of the figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    size = map_obj.size
    
    # Draw grid
    for i in range(size+1):
        ax.axhline(i, color='gray', linewidth=0.5)
        ax.axvline(i, color='gray', linewidth=0.5)
    
    # Draw walls
    walls = map_obj.get_walls()
    for y in range(size):
        for x in range(size):
            # Right wall (x, y) -> (x+1, y) to (x+1, y+1)
            if walls[y, x, 0]:
                ax.plot([x+1, x+1], [y, y+1], 'k-', linewidth=2)
            # Bottom wall (x, y+1) -> (x+1, y+1)
            if walls[y, x, 1]:
                ax.plot([x, x+1], [y+1, y+1], 'k-', linewidth=2)
            # Left wall (x, y) -> (x, y+1)
            if walls[y, x, 2]:
                ax.plot([x, x], [y, y+1], 'k-', linewidth=2)
            # Top wall (x, y) -> (x+1, y)
            if walls[y, x, 3]:
                ax.plot([x, x+1], [y, y], 'k-', linewidth=2)
    
    # Draw tile types
    tile_type = map_obj.get_tile_type()
    if show_points:
        for y in range(size):
            for x in range(size):
                if tile_type[y, x] == map_obj.RECON:
                    ax.add_patch(Circle((x + 0.5, y + 0.5), 0.2, color='blue', alpha=0.5))
                elif tile_type[y, x] == map_obj.MISSION:
                    ax.add_patch(Circle((x + 0.5, y + 0.5), 0.3, color='green', alpha=0.7))
    
    # Draw agents
    if show_agents:
        scouts, guards = map_obj.get_agents()
        for y in range(size):
            for x in range(size):
                if scouts[y, x]:
                    ax.add_patch(Rectangle((x + 0.25, y + 0.25), 0.5, 0.5, color='cyan', alpha=0.7))
                if guards[y, x]:
                    ax.add_patch(Rectangle((x + 0.25, y + 0.25), 0.5, 0.5, color='red', alpha=0.7))
    
    # Draw visited locations
    if show_visited:
        visited = map_obj.get_visited()
        for y in range(size):
            for x in range(size):
                if visited[y, x]:
                    ax.add_patch(Rectangle((x, y), 1, 1, color='yellow', alpha=0.1))
    
    # Set limits and labels
    ax.set_xlim(0, size)
    ax.set_ylim(0, size)
    ax.set_xticks(np.arange(0.5, size))
    ax.set_yticks(np.arange(0.5, size))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.invert_yaxis()  # Invert y-axis to match the grid coordinate system
    
    # Legend
    legend_elements = [
        Rectangle((0, 0), 1, 1, color='yellow', alpha=0.1, label='Visited'),
        Rectangle((0, 0), 1, 1, color='cyan', alpha=0.7, label='Scout'),
        Rectangle((0, 0), 1, 1, color='red', alpha=0.7, label='Guard'),
        Circle((0, 0), 0.2, color='blue', alpha=0.5, label='Recon (1 pt)'),
        Circle((0, 0), 0.3, color='green', alpha=0.7, label='Mission (5 pts)')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
    
    plt.title('Reconstructed Map')
    plt.tight_layout()
    return fig, ax

def save_map_visualization(map_obj, filename='map.png', **kwargs):
    """Save the map visualization to a file"""
    fig, _ = visualize_map(map_obj, **kwargs)
    plt.savefig(filename)
    plt.close(fig)
    return filename

def show_map(map_obj, **kwargs):
    """Show the map visualization"""
    visualize_map(map_obj, **kwargs)
    plt.show()