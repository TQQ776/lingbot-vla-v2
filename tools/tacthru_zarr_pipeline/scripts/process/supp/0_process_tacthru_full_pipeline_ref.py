import glob
import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
import torch.multiprocessing as mp
from loguru import logger
from tqdm.rich import tqdm_rich as tqdm

from utils.proc_utils import KeypointsKFProcessor
from t4.wds_utils import WdsWriter

force_reinpaint = False
force_detect = False

ds_rate = 1
results = defaultdict(list)

job_lists = {
    "tacthru_m": {
        "data_list": [
            "data/tacthru_raw/PickBottle-v4/demos/*/TacThru-0_synced.mp4",
            "data/tacthru_raw/PullTissue-v1/demos/*/TacThru-0_synced.mp4",
            "data/tacthru_raw/Scissors-v3/demos/*/TacThru-0_synced.mp4",
            "data/tacthru_raw/SortBolt-v5/demos/*/TacThru-0_synced.mp4",
            "data/tacthru_raw/InsertCap-v6/demos/*/TacThru-0_synced.mp4",
        ],
        "detector": "double_det",
        "depth_rel_path": "depth/TacThru-0_synced_inpainted_depthanything.mp4",
        "inpainted_rel_path": "demos/*/TacThru-0_synced_inpainted.mp4",
    }
}


def process_single_video(args):
    """Process a single video and return the results"""
    video_path, sensor_name, detector_type, ds_rate, do_inpaint = args
    logger.debug(f"Started processing {video_path}")

    video = cv2.VideoCapture(video_path)
    output_inpainted_video_writer = None

    tgt_video_path = None
    if do_inpaint:
        tgt_video_path = output_inpainted_video_path = video_path.replace(".avi", "_inpainted.avi").replace(".mp4", "_inpainted.mp4")
        if force_reinpaint or not os.path.exists(output_inpainted_video_path):
            output_inpainted_video_writer = cv2.VideoWriter(
                output_inpainted_video_path,
                cv2.VideoWriter_fourcc(*"XVID"),
                video.get(cv2.CAP_PROP_FPS),
                (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))),
            )
            logger.debug(f"Dumping new inpainted sensor video to {output_inpainted_video_path}")

    kpts_cache_path = video_path.replace(".avi", "_kpts.pkl").replace(".mp4", "_kpts.pkl")
    video_labels = {"marker_ref": [], "marker": [], "marker_flow": []}

    detector = None
    if not force_detect and os.path.exists(kpts_cache_path):
        with open(kpts_cache_path, "rb") as f:
            video_labels = pickle.load(f)
    else:
        ref_marker_pos_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(video_path))), "assets", "tacthru", "ref_kpts.npy")
        tacthru_ref_marker_pos = np.load(ref_marker_pos_path) * 400
        detector = KeypointsKFProcessor(tacthru_ref_marker_pos)

        detector.reset()

    i_frame = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
        frame_shape = frame.shape[:2][::-1]  # (width, height)

        if i_frame % ds_rate != 0:
            continue

        if detector is None:
            kpts = (np.array(video_labels["marker"][i_frame]) + 1) / 2 * np.array(frame_shape)
            kpts_ref = (np.array(video_labels["marker_ref"][i_frame]) + 1) / 2 * np.array(frame_shape)
        else:
            res_dict = detector(frame)
            kpts, kpts_ref = res_dict["marker"], res_dict["marker_ref"]

        if output_inpainted_video_writer is not None:
            frame_inpaint = frame.copy()
            marker_inpaint_radius = 12
            inpaint_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for kpt in kpts:
                cv2.circle(inpaint_mask, kpt.astype(int), marker_inpaint_radius, (255, 255, 255), cv2.FILLED)
            # cv2.imwrite("debug_mask.png", inpaint_mask)

            frame_inpaint = cv2.inpaint(frame_inpaint, inpaint_mask, 5, cv2.INPAINT_TELEA)
            output_inpainted_video_writer.write(frame_inpaint)

        if detector is not None:
            kpts = (kpts / np.array(frame_shape)).astype(np.float32) * 2 - 1
            kpts_ref = (kpts_ref / np.array(frame_shape)).astype(np.float32) * 2 - 1

            assert len(kpts) == 64, f"Expected 64 markers, got {len(kpts)} in {video_path} at frame {i_frame}"

            video_labels["marker_ref"].append(kpts_ref.tolist())
            video_labels["marker"].append(kpts.tolist())
            video_labels["marker_flow"].append((kpts - kpts_ref).tolist())
        i_frame += 1

    if output_inpainted_video_writer is not None:
        output_inpainted_video_writer.release()
    video.release()

    if detector is not None:
        with open(kpts_cache_path, "wb") as f:
            pickle.dump(video_labels, f)
        logger.debug(f"Saved kpts cache to {kpts_cache_path}")

    return video_path, tgt_video_path, video_labels


for sensor_name, sensor_job in job_lists.items():
    logger.info(f"Processing sensor: {sensor_name}")

    data_list = sensor_job["data_list"]
    depth_rel_path = sensor_job.get("depth_rel_path")

    wds_writer = WdsWriter(50000, 5000)
    export_dir = os.path.join("data/t4/tacthru_m", f"{sensor_name}")
    os.makedirs(export_dir, exist_ok=True)

    all_videos = []
    for f in data_list:
        files = glob.glob(f)
        assert len(files) > 0, f"No files found for pattern: {f}"
        all_videos.extend(files)
    logger.info(f"Found {len(all_videos)} videos for {sensor_name} sensor")

    video_labels = defaultdict(list)
    depth_paths = []

    # Prepare arguments for parallel processing
    if (detector_type := sensor_job.get("detector")) is not None:
        process_args = [(video_path, sensor_name, detector_type, ds_rate, "inpainted_rel_path" in sensor_job) for video_path in all_videos]

        # Process videos in parallel
        with mp.Pool(processes=mp.cpu_count() // 2) as pool:
            results = list(tqdm(pool.imap(process_single_video, process_args), total=len(all_videos)))

        # Collect results
        raw_video_paths, tgt_video_paths = [], []
        for raw_video_path, tgt_video_path, labels in results:
            raw_video_paths.append(raw_video_path)
            tgt_video_paths.append(tgt_video_path)

            # Merge labels from all videos
            for key, values in labels.items():
                video_labels[key].extend(values)

    else:
        raw_video_paths = all_videos

    for raw_video_path in raw_video_paths:
        if depth_rel_path is not None:
            depth_path = os.path.join(os.path.dirname(raw_video_path), depth_rel_path)
            depth_paths.append(depth_path)

    video_names, depth_video_names = [], []
    for raw_video_path in raw_video_paths:
        task_name = os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(raw_video_path))))
        run_name = os.path.basename(os.path.dirname(raw_video_path))
        video_name = f"{task_name}_{run_name}_{sensor_name}"
        logger.info(f"Processed {task_name} > {run_name} > {sensor_name}")
        video_names.append(video_name)
        depth_video_names.append(video_name + "-depth")

    assert len(np.unique(video_names)) == len(video_names), "Video names are not unique!"

    target_size = (224, 224)

    wds_writer.add_video_labels("rgb.jpg", raw_video_paths, ds_rate, target_size, video_names)
    wds_writer.add_video_labels("rgb_tgt.jpg", tgt_video_paths, ds_rate, target_size, video_names)
    # if len(depth_paths) > 0:
    assert len(depth_paths) > 0
    wds_writer.add_video_labels("depth.jpg", depth_paths, ds_rate, target_size, depth_video_names)
    wds_writer.add_labels(**video_labels)

    wds_writer.save(export_dir, val_ratio=0.15, video_n_workers=mp.cpu_count() // 2)
