from argparse import ArgumentParser
import json
from tqdm import tqdm
import numpy as np
import re
import os.path
import subprocess
import traceback
from copy import deepcopy
from utils import get_box
from utils import stack_all_pids
from utils import best_matching_hungarian


def sort_serial_number_filenames(list_to_sort):
    '''Sort filenames that are serial numbers.'''
    match_list = [(re.search("[0-9]+", x).group(), x) for x in list_to_sort]
    match_list.sort(cmp = lambda x, y: cmp(int(x[0]), int(y[0])))
    
    return [x[1] for x in match_list]


def read_or_generate_deepmatching_result(frame_id_1, frame_id_2, video_dir, deepmatching_exec_path='./deepmatching/deepmatching'):
    cor_filepath = '{}/deepmatching_result/{}_{}.txt'.format(video_dir, frame_id_1, frame_id_2)

    # Just read cor_filepath if it exists
    if os.path.exists(cor_filepath):
        return np.loadtxt(cor_filepath)
    # Generate pair-matching txt if not exists
    else:
        img1_path = '{}/frames/{}.jpg'.format(video_dir, frame_id_1)
        img2_path = '{}/frames/{}.jpg'.format(video_dir, frame_id_2)

        # Execute deepmatching
        cmd = '{} {} {} -nt 20 -downscale 2 -out {}'.format(deepmatching_exec_path, img1_path, img2_path, cor_filepath)
        try:
            subprocess.call([cmd], shell=True)
        except OSError:
            print('Failed on executing deepmatching with command: {}'.format(cmd))
            traceback.print_exc()
        finally:
            return np.loadtxt(cor_filepath)


def load_pose_data(json_filename, video_dir):
    print('Loading pose data...')

    track = {}

    with open(json_filename, 'r') as f:
        pose_data = json.load(f)

        for single_pose_data in tqdm(pose_data):
            # get image id
            image_id = single_pose_data['image_id']
            # make new dict if image not exist
            if image_id not in track.keys():
                track[image_id] = {}

            # get number of boxes
            number_of_boxes = len(track[image_id].keys())
            # make new dict of box
            track[image_id][number_of_boxes + 1] = {}
            track[image_id][number_of_boxes + 1]['box_score'] = single_pose_data['score']
            track[image_id][number_of_boxes + 1]['box_pos'] = get_box(single_pose_data['keypoints'], '{}/frames/{}'.format(video_dir, image_id))
            track[image_id][number_of_boxes + 1]['box_pose_pos'] = np.array(single_pose_data['keypoints']).reshape(-1,3)[:,0:2].tolist()
            track[image_id][number_of_boxes + 1]['box_pose_score'] = np.array(single_pose_data['keypoints']).reshape(-1,3)[:,-1].tolist()

        # Get number of boxes of each frame and creat a field for that data
        for image_id in track.keys():
            track[image_id]['num_boxes'] = len(track[image_id])

    print('---> Done.')
    return track


def main():
    parser = ArgumentParser(description='FoseFlow Tracker')
    parser.add_argument('--video_dir', type=str)
    parser.add_argument('--pose_data_path', type=str)
    parser.add_argument('--link', type=int, default=100, help='look-ahead LINK_LEN frames to find tracked human bbox')
    parser.add_argument('--num', type=int, default=7, help='pick high-score(top NUM) keypoints when computing pose_IOU (?)')
    parser.add_argument('--mag', type=int, default=30, help='box width/height around keypoint for computing pose IoU (?)')
    parser.add_argument('--match', type=float, default=0.2, help='match threshold in Hungarian Matching')
    args = parser.parse_args()

    # Load pose data and initialize tracking data
    track = load_pose_data(args.pose_data_path, args.video_dir)
    # Sort frame list of serial numbers
    frame_list = sort_serial_number_filenames(track.keys())
    # Initialize maximum number of people tracked through the video (I think and not very sure!!)
    max_pid_id = 0
    # Initialize weights
    weights = [1,0,1,0,0,0] 
    weights_fff = [0,1,0,1,0,0]

    # For all frames
    print('Tracking pose data...')
    for idx, frame_name in enumerate(tqdm(frame_list[:-1])): # Loop until the second last frame
        # Get current frame id
        frame_id = frame_name.split(".")[0]

        # Get next frame id and name
        next_frame_name = frame_list[idx+1]
        next_frame_id = next_frame_name.split(".")[0]

        # Initialize tracking info of the first frame
        if idx == 0:
            for pid in range(1, track[frame_name]['num_boxes'] + 1):
                    track[frame_name][pid]['new_pid'] = pid
                    track[frame_name][pid]['match_score'] = 0

        # Update maximum number
        max_pid_id = max(max_pid_id, track[frame_name]['num_boxes'])

        # Read or generate deepmatching result
        all_cors = read_or_generate_deepmatching_result(frame_id, next_frame_id, args.video_dir)

        # if there is no people in this frame, then copy the info from former frame
        if len(track[next_frame_name]) == 0:
            track[next_frame_name] = deepcopy(track[frame_name])
            continue

        # Get matches with hungarian
        cur_all_pids, cur_all_pids_fff = stack_all_pids(track, frame_list[:-1], idx, max_pid_id, args.link)
        match_indexes, match_scores = best_matching_hungarian(
            all_cors, 
            cur_all_pids, 
            cur_all_pids_fff, 
            track[next_frame_name], 
            weights, 
            weights_fff, 
            args.num, 
            args.mag
        )

        # Add match result when match_score > threshold
        for pid1, pid2 in match_indexes:
            if match_scores[pid1][pid2] > args.match:
                track[next_frame_name][pid2 + 1]['new_pid'] = cur_all_pids[pid1]['new_pid']
                max_pid_id = max(max_pid_id, track[next_frame_name][pid2 + 1]['new_pid'])
                track[next_frame_name][pid2 + 1]['match_score'] = match_scores[pid1][pid2]

        # Add the untracked new person
        for next_pid in range(1, track[next_frame_name]['num_boxes'] + 1):
            if 'new_pid' not in track[next_frame_name][next_pid]:
                max_pid_id += 1
                track[next_frame_name][next_pid]['new_pid'] = max_pid_id
                track[next_frame_name][next_pid]['match_score'] = 0

    print('---> Done.')

    # Save result to json file
    output_filepath = '{}/poseflow_result.json'.format(args.video_dir)
    with open(output_filepath, 'w') as fp:
        json.dump(track, fp, indent=2, sort_keys=True)
        print('Saved result json to: {}'.format(output_filepath))



if __name__ == '__main__':
    main()
