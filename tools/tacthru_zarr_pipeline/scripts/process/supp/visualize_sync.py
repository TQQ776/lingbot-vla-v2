import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation as R
import os
import argparse
from pathlib import Path


class SynchronizedVisualizer:
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.poses = None
        self.ts = None
        self.cap = None
        self.fig = None
        self.ax3d = None
        self.ax_video = None
        self.current_frame = 0
        self.total_frames = 0

        # Load data
        self.load_pose_data()
        self.load_video_data()

    def load_pose_data(self):
        """Load and process pose data from CSV"""
        csv_path = self.data_dir / "vive_synced.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Could not find {csv_path}")

        # Load CSV data
        df = pd.read_csv(csv_path)
        self.poses = df[["timestamp", "x", "y", "z", "q_x", "q_y", "q_z", "q_w"]].copy()

        self.ts = np.array(self.poses["timestamp"].values)
        # Transform poses to first frame coordinate system
        self.transform_to_first_frame()

    def transform_to_first_frame(self):
        """Transform all poses to the coordinate system of the first frame"""
        if len(self.poses) == 0:
            return

        # Get first pose
        first_pos = self.poses.iloc[0][["x", "y", "z"]].values
        first_quat = self.poses.iloc[0][["q_x", "q_y", "q_z", "q_w"]].values
        first_rot = R.from_quat(first_quat)

        # Transform all poses
        transformed_positions = []
        transformed_rotations = []

        for idx, row in self.poses.iterrows():
            # Current pose
            pos = row[["x", "y", "z"]].values
            quat = row[["q_x", "q_y", "q_z", "q_w"]].values
            rot = R.from_quat(quat)

            # Transform to first frame coordinate system
            # T_first_current = T_first_world * T_world_current
            # Since we want T_first_current, we compute: T_first_world^-1 * T_world_current
            relative_pos = first_rot.inv().apply(pos - first_pos)
            relative_rot = first_rot.inv() * rot

            transformed_positions.append(relative_pos)
            transformed_rotations.append(relative_rot)

        self.poses["x_rel"] = [pos[0] for pos in transformed_positions]
        self.poses["y_rel"] = [pos[1] for pos in transformed_positions]
        self.poses["z_rel"] = [pos[2] for pos in transformed_positions]
        self.poses["rot_obj"] = transformed_rotations

    def load_video_data(self):
        """Load video and corresponding timestamps"""
        video_path = self.data_dir / "camera_synced.mp4"

        if not video_path.exists():
            raise FileNotFoundError(f"Could not find {video_path}")

        # Load video
        self.cap = cv2.VideoCapture(str(video_path))
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video {video_path}")

        self.total_frames = len(self.ts)
        print(f"Loaded video with {self.total_frames} frames")
        print(f"Loaded {len(self.poses)} pose samples")

    def find_closest_pose(self, video_timestamp):
        """Find the pose closest to the given video timestamp"""
        if len(self.poses) == 0:
            return None

        # Find closest timestamp
        time_diffs = np.abs(self.poses["timestamp"] - video_timestamp)
        closest_idx = time_diffs.idxmin()
        return self.poses.iloc[closest_idx]

    def draw_coordinate_axes(self, ax, position, rotation, scale=0.05):
        """Draw coordinate axes at given position and rotation"""
        # Define axis vectors
        axes_vectors = np.array(
            [
                [scale, 0, 0],  # X-axis (red)
                [0, scale, 0],  # Y-axis (green)
                [0, 0, scale],
            ]
        )  # Z-axis (blue)

        # Rotate axes vectors
        rotated_axes = rotation.apply(axes_vectors)

        # Colors for X, Y, Z axes
        colors = ["red", "green", "blue"]
        labels = ["X", "Y", "Z"]

        for i, (axis, color, label) in enumerate(zip(rotated_axes, colors, labels)):
            end_point = position + axis
            ax.plot(
                [position[0], end_point[0]],
                [position[1], end_point[1]],
                [position[2], end_point[2]],
                color=color,
                linewidth=2,
                label=f"{label}-axis" if i == 0 else "",
            )

    def setup_plot(self):
        """Setup the matplotlib figure and axes"""
        self.fig = plt.figure(figsize=(15, 8))

        # 3D plot for trajectory
        self.ax3d = self.fig.add_subplot(121, projection="3d")
        self.ax3d.set_title("3D Trajectory (Relative to First Frame)")
        self.ax3d.set_xlabel("X (m)")
        self.ax3d.set_ylabel("Y (m)")
        self.ax3d.set_zlabel("Z (m)")

        # Video display
        self.ax_video = self.fig.add_subplot(122)
        self.ax_video.set_title("Synchronized Video")
        self.ax_video.axis("off")

        # Plot full trajectory
        if len(self.poses) > 0:
            self.ax3d.plot(self.poses["x_rel"], self.poses["y_rel"], self.poses["z_rel"], "b-", alpha=0.3, label="Full trajectory")

        # Set equal aspect ratio for 3D plot
        self.set_3d_equal_aspect()

    def set_3d_equal_aspect(self):
        """Set equal aspect ratio for 3D plot"""
        if len(self.poses) == 0:
            return

        x_data = self.poses["x_rel"]
        y_data = self.poses["y_rel"]
        z_data = self.poses["z_rel"]

        max_range = np.array([x_data.max() - x_data.min(), y_data.max() - y_data.min(), z_data.max() - z_data.min()]).max() / 2.0

        mid_x = (x_data.max() + x_data.min()) * 0.5
        mid_y = (y_data.max() + y_data.min()) * 0.5
        mid_z = (z_data.max() + z_data.min()) * 0.5

        self.ax3d.set_xlim(mid_x - max_range, mid_x + max_range)
        self.ax3d.set_ylim(mid_y - max_range, mid_y + max_range)
        self.ax3d.set_zlim(mid_z - max_range, mid_z + max_range)

    def animate(self, frame_idx):
        """Animation function for matplotlib"""
        if frame_idx >= self.total_frames:
            return

        # Get current video timestamp
        current_time = self.ts[frame_idx]

        # Find corresponding pose
        closest_pose = self.find_closest_pose(current_time)

        # Read video frame
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, video_frame = self.cap.read()

        if ret:
            # Convert BGR to RGB for matplotlib
            video_frame = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
            self.ax_video.clear()
            self.ax_video.imshow(video_frame)
            self.ax_video.set_title(f"Frame {frame_idx}/{self.total_frames}\nTime: {current_time:.3f}s")
            self.ax_video.axis("off")

        # Update 3D plot
        self.ax3d.clear()

        # Redraw trajectory
        if len(self.poses) > 0:
            self.ax3d.plot(self.poses["x_rel"], self.poses["y_rel"], self.poses["z_rel"], "b-", alpha=0.3, label="Full trajectory")

        # Draw current pose if available
        if closest_pose is not None:
            position = np.array([closest_pose["x_rel"], closest_pose["y_rel"], closest_pose["z_rel"]])
            rotation = closest_pose["rot_obj"]

            # Draw coordinate axes at current pose
            self.draw_coordinate_axes(self.ax3d, position, rotation)

            # Highlight current position
            self.ax3d.scatter([position[0]], [position[1]], [position[2]], c="red", s=50, label="Current pose")

        self.ax3d.set_title("3D Trajectory (Relative to First Frame)")
        self.ax3d.set_xlabel("X (m)")
        self.ax3d.set_ylabel("Y (m)")
        self.ax3d.set_zlabel("Z (m)")
        self.ax3d.legend()

        # Maintain equal aspect ratio
        self.set_3d_equal_aspect()

    def run_visualization(self, fps=30, export_video=False):
        """Run the synchronized visualization"""
        self.setup_plot()

        # Calculate interval for desired FPS
        interval = 1000 / fps  # milliseconds

        # Create animation
        frames = min(self.total_frames, 1000) if not export_video else self.total_frames
        anim = FuncAnimation(self.fig, self.animate, frames=frames, interval=interval, repeat=True, blit=False)

        if export_video:
            # Export as MP4 video
            output_path = self.data_dir / "trajectory.mp4"
            print(f"Exporting video to {output_path} at {fps} FPS...")

            # Setup writer
            writer = FFMpegWriter(fps=fps, metadata=dict(artist="SynchronizedVisualizer"), bitrate=5000)

            # Save animation
            anim.save(str(output_path), writer=writer, progress_callback=self._progress_callback)
            print(f"Video exported successfully to {output_path}")

            # Close figure to free memory
            plt.close(self.fig)
        else:
            plt.tight_layout()
            plt.show()

        return anim

    def _progress_callback(self, current_frame, total_frames):
        """Progress callback for video export"""
        progress = (current_frame / total_frames) * 100
        if current_frame % 30 == 0:  # Print progress every 30 frames
            print(f"Progress: {progress:.1f}% ({current_frame}/{total_frames} frames)")

    def __del__(self):
        """Cleanup"""
        if self.cap:
            self.cap.release()


def main():
    parser = argparse.ArgumentParser(description="Visualize synchronized 3D trajectory and video")
    parser.add_argument("data_dir", type=str, help="Directory containing vive_synced.csv and camera_synced.mp4")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
    parser.add_argument("--export", action="store_true", help="Export as trajectory.mp4 instead of showing interactive visualization")

    args = parser.parse_args()

    try:
        visualizer = SynchronizedVisualizer(args.data_dir)

        if args.export:
            anim = visualizer.run_visualization(fps=args.fps, export_video=True)
        else:
            # Show interactive visualization
            anim = visualizer.run_visualization(fps=args.fps)
            plt.show()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
