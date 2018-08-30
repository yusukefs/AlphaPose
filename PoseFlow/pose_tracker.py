from argparse import ArgumentParser
import json
from tqdm import tqdm
import numpy as np
from utils import get_box


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
            track[image_id][number_of_boxes + 1]['box_pose_pos'] = np.array(single_pose_data['keypoints']).reshape(-1,3)[:,0:2]
            track[image_id][number_of_boxes + 1]['box_pose_score'] = np.array(single_pose_data['keypoints']).reshape(-1,3)[:,-1]

    print('---> Done.')
    return track


def main():
    parser = ArgumentParser(description='FoseFlow Tracker')
    parser.add_argument('--video_dir', type=str)
    parser.add_argument('--pose_data_path', type=str)
    args = parser.parse_args()

    load_pose_data(args.pose_data_path, args.video_dir)


if __name__ == '__main__':
    main()
