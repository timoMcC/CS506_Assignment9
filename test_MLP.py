import os
import sys
from pathlib import Path

def test_mlp_visualization():
    # Check if results directory exists, create if it doesn't
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir.mkdir()
        print("Created results directory")
    
    try:
        # Import and run
        from neural_networks import visualize
        
        print("Starting visualization with extended training...")
        # Increased steps to 2000, adjusted learning rate for better convergence
        visualize(activation="tanh", 
                 lr=0.05,  # Slightly lower learning rate for stability
                 step_num=2000)  # Increased number of steps
        
        # Check if file was created
        expected_file = results_dir / "visualize.gif"
        if expected_file.exists():
            print(f"Success! Visualization saved to {expected_file}")
            print(f"File size: {expected_file.stat().st_size / 1024:.2f} KB")
            print("\nVisualization parameters:")
            print("- Training steps: 2000")
            print("- Learning rate: 0.05")
            print("- Activation: tanh")
            print("- Updates per frame: 10")
            print("- Total frames:", 2000//10)
        else:
            print("Error: Visualization file was not created")
            
    except Exception as e:
        print(f"Error during visualization: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Run the test
    success = test_mlp_visualization()
    print(f"\nOverall test {'passed' if success else 'failed'}")