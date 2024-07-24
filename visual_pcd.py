import open3d as o3d
from tkinter import Tk, filedialog, messagebox
import os

def main():
    try:
        # Hide the root tkinter window
        root = Tk()
        root.withdraw()

        # Open a file dialog to select the point cloud file
        last_dir = os.getcwd()  # Default to current working directory
        point_path = filedialog.askopenfilename(
            title="Select point cloud file",
            initialdir=last_dir,
            filetypes=[("PCD files", "*.pcd"), ("All files", "*.*")]
        )

        if point_path:
            print(f"Selected file: {point_path}")

            try:
                pcd = o3d.io.read_point_cloud(point_path)
                if pcd.is_empty():
                    raise ValueError("The selected point cloud file is empty.")
                o3d.visualization.draw_geometries([pcd])
                print("Point cloud visualization completed successfully.")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load point cloud: {e}")
                print(f"Error loading point cloud: {e}")
        else:
            print("No file selected.")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred: {e}")
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()
