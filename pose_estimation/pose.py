# pose.py: conducts 2D pose detection and lifting to 3D

from easydict import EasyDict
from alphapose.models import builder
from alphapose.utils.bbox import _box_to_center_scale, _center_scale_to_box
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.transforms import get_affine_transform, im_to_torch, get_func_heatmap_to_coord
from alphapose.utils.vis import vis_frame_fast as vis_frame

from torch.utils.data import DataLoader
from .motionbert.lib.utils.tools import *
from .motionbert.lib.utils.learning import *
from .motionbert.lib.utils.utils_data import flip_data

import sys, tqdm, cv2, os, json, yaml, numpy, torch

# get the forward direction of a character
# pose: list of keypoints
def get_direction_from_pose(pose:list, use_gt_pose=False) -> list:
    # get direction from ground truth pose
    if use_gt_pose:
        return pose[-1]  # use the coordinate system (east, north, vertical)
    # get direction from observed pose
    else:
        print("POSE DIRECTION FROM NON-GT IS NOT IMPLEMENTED")
        return pose[-1]  # use the coordinate system (east, north, vertical)

class PoseDetector(object):
    def __init__(self, config_path:str="", checkpoint_path:str="", device:str="cpu"):
        # load the config
        cfg_path = "pose_estimation/AlphaPose/256x192_res50_lr1e-3_2x.yaml" if config_path == "" else config_path
        with open(cfg_path) as f:
            self.pose_model_cfg = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

        self.pose_model_args = EasyDict()
        self.pose_model_args.checkpoint = "pose_estimation/AlphaPose/halpe26_fast_res50_256x192.pth" if checkpoint_path == "" else checkpoint_path
        self.pose_model_args.device = device
        self.pose_model_args.gpus = [0]
        self.pose_model_args.qsize = 100
        self.pose_model_args.outputpath = "./"
        self.pose_model_args.tracking = False
        self.pose_model_args.showbox = True
        self.pose_model_args.posebatch = 10
        self.pose_model_args.min_box_area = 30000

        self.pose_model = builder.build_sppe(self.pose_model_cfg.MODEL, preset_cfg=self.pose_model_cfg.DATA_PRESET)
        self.pose_model.load_state_dict(torch.load(self.pose_model_args.checkpoint, map_location=self.pose_model_args.device, weights_only=True))
        self.pose_model.to(self.pose_model_args.device)
        self.pose_model.eval()
        print("Pose model loaded successfully")

    def get_config(self, config_path):
        yaml.add_constructor('!include', construct_include, Loader)
        with open(config_path, 'r') as stream:
            config = yaml.load(stream, Loader=Loader)
        config = edict(config)
        _, config_filename = os.path.split(config_path)
        config_name, _ = os.path.splitext(config_filename)
        config.name = config_name
        return config

    # get the 3D pose from an image with a bounding box
    def get_3d_pose(self, image, box, keypoints_2d):
        print("IMAGE", image.shape)
        config = "pose_estimation/motionbert/configs/pose3d/MB_ft_h36m_global_lite.yaml"
        checkpoint = "pose_estimation/motionbert/checkpoint/pose3d/FT_MB_lite_MB_ft_h36m_global_lite/best_epoch.bin"
        args = get_config(config)

        model_backbone = load_backbone(args)
        if torch.cuda.is_available():
            model_backbone = nn.DataParallel(model_backbone)
            model_backbone = model_backbone.cuda()

        print('Loading checkpoint', checkpoint)

        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage, weights_only=True)["model_pos"]
        checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}

        model_backbone.load_state_dict(checkpoint, strict=True)
        model_pos = model_backbone
        model_pos.eval()
        testloader_params = {
                'batch_size': 1,
                'shuffle': False,
                'num_workers': 8,
                'pin_memory': True,
                'prefetch_factor': 4,
                'persistent_workers': True,
                'drop_last': False
        }

        sequence_length = 16
        image = torch.Tensor(image).permute(2, 0, 1)  # make the image a Tensor and change the image dimensions from H W C to C H W
        frame_sequence = torch.zeros(sequence_length, 3, image.size(1), image.size(2))  # Zero pad
        frame_sequence[0] = image  # Insert the actual frame in the first position
        frame_sequence = frame_sequence.unsqueeze(0)  # Add batch dimension: (1, seq_len, C, H, W)

        input_frames = torch.zeros(1, sequence_length, len(keypoints_2d[0]["keypoints"]), 2)
        input_frames[0,0,:,:] = torch.Tensor(keypoints_2d[0]["keypoints"])

        # Pass this into the model
        print("INPUT", input_frames.shape)
        output = model_backbone(input_frames)  # takes in a tensor of shape B F J C -- B is batch size, F is the number of frames, J is the number of joints, C is the number of dimensions per joint (2, 3)

        print("OUTPUT", output)

        vid = imageio.get_reader(opts.vid_path, 'ffmpeg')
        fps_in = vid.get_meta_data()['fps']
        vid_size = vid.get_meta_data()['size']
        os.makedirs(opts.out_path, exist_ok=True)

        if opts.pixel:
            # Keep relative scale with pixel coornidates
            wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, vid_size=vid_size, scale_range=None, focus=opts.focus)
        else:
            # Scale to [-1,1]
            wild_dataset = WildDetDataset(opts.json_path, clip_len=opts.clip_len, scale_range=[1,1], focus=opts.focus)

        test_loader = DataLoader(wild_dataset, **testloader_params)

        results_all = []
        with torch.no_grad():
            for batch_input in tqdm(test_loader):
                N, T = batch_input.shape[:2]
                if torch.cuda.is_available():
                    batch_input = batch_input.cuda()
                if args.no_conf:
                    batch_input = batch_input[:, :, :, :2]
                if args.flip:
                    batch_input_flip = flip_data(batch_input)
                    predicted_3d_pos_1 = model_pos(batch_input)
                    predicted_3d_pos_flip = model_pos(batch_input_flip)
                    predicted_3d_pos_2 = flip_data(predicted_3d_pos_flip) # Flip back
                    predicted_3d_pos = (predicted_3d_pos_1 + predicted_3d_pos_2) / 2.0
                else:
                    predicted_3d_pos = model_pos(batch_input)
                if args.rootrel:
                    predicted_3d_pos[:,:,0,:]=0                    # [N,T,17,3]
                else:
                    predicted_3d_pos[:,0,0,2]=0
                    pass
                if args.gt_2d:
                    predicted_3d_pos[...,:2] = batch_input[...,:2]
                results_all.append(predicted_3d_pos.cpu().numpy())

        results_all = np.hstack(results_all)
        results_all = np.concatenate(results_all)
        render_and_save(results_all, '%s/X3D.mp4' % (opts.out_path), keep_imgs=False, fps=fps_in)
        if opts.pixel:
            # Convert to pixel coordinates
            results_all = results_all * (min(vid_size) / 2.0)
            results_all[:,:,:2] = results_all[:,:,:2] + np.array(vid_size) / 2.0
        np.save('%s/X3D.npy' % (opts.out_path), results_all)
        return

    # crop the image around boxes
    def __crop_around_boxes_2d__(self, image, boxes):
        processed_data = []
        with torch.no_grad():
            scores = torch.ones(boxes.size(0))
            ids = torch.Tensor(range(boxes.size(0)))
            inps = torch.zeros(boxes.size(0), 3, *self.pose_model_cfg.DATA_PRESET.IMAGE_SIZE)
            cropped_boxes = torch.zeros(boxes.size(0), 4)
            for i, box in enumerate(boxes):
                xmin, ymin, xmax, ymax = box
                inp_h, inp_w = self.pose_model_cfg.DATA_PRESET.IMAGE_SIZE
                center, scale = _box_to_center_scale(xmin, ymin, xmax - xmin, ymax - ymin, float(inp_w) / inp_h)
                scale = scale * 1.0
                trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])
                img = cv2.warpAffine(image, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
                bbox = _center_scale_to_box(center, scale)
                inps[i] = im_to_torch(img)
                inps[i][0].add_(-0.406)
                inps[i][1].add_(-0.457)
                inps[i][2].add_(-0.480)
                cropped_boxes[i] = torch.FloatTensor(bbox)
        return (inps, scores, ids, cropped_boxes)

    # draw the poses on the image
    def draw_poses_2d(self, image:numpy.ndarray, keypoints:dict):
        return vis_frame(image, keypoints, self.pose_model_args, [0.4] * len(keypoints[0]["keypoints"]))

    # write the 2D pose results to a json file
    def write_json_2d(self, result, outputpath, outputfile='alphapose-results.json'):
        json_results = []
        json_results_cmu = {}
        im_name = result['imgname']
        for human in result['result']:
            keypoints = []
            result = {}
            result['image_id'] = os.path.basename(im_name)
            result['category_id'] = 1
            for n in range(human['kp_score'].shape[0]):
                keypoints.append(float(human['keypoints'][n, 0]))
                keypoints.append(float(human['keypoints'][n, 1]))
                keypoints.append(float(human['kp_score'][n]))
            result['keypoints'] = keypoints
            result['score'] = float(human['proposal_score'])
            result['box'] = human['box']
            result['idx'] = human['idx']
            json_results.append(result)
        with open(os.path.join(outputpath, outputfile), 'w') as json_file:
            json_file.write(json.dumps(json_results))

    # convert the heatmap to a result dictionary
    def __heatmap_to_result_dict_2d__(self, image_and_boxes=None) -> dict[str, dict]:
        final_result = []
        norm_type = self.pose_model_cfg.LOSS.get('NORM_TYPE', None)
        hm_size = self.pose_model_cfg.DATA_PRESET.HEATMAP_SIZE
        (boxes, scores, ids, hm_data, cropped_boxes, orig_img) = image_and_boxes
        orig_img = numpy.array(orig_img, dtype=numpy.uint8)[:, :, ::-1]
        assert hm_data.dim() == 4

        face_hand_num = 110
        if hm_data.size()[1] == 136:
            eval_joints = [*range(0,136)]
        elif hm_data.size()[1] == 26:
            eval_joints = [*range(0,26)]
        elif hm_data.size()[1] == 133:
            eval_joints = [*range(0,133)]
        elif hm_data.size()[1] == 68:
            face_hand_num = 42
            eval_joints = [*range(0,68)]
        elif hm_data.size()[1] == 21:
            eval_joints = [*range(0,21)]
        pose_coords = []
        pose_scores = []
        for i in range(hm_data.shape[0]):
            bbox = cropped_boxes[i].tolist()
            pose_coord, pose_score = get_func_heatmap_to_coord(self.pose_model_cfg)(hm_data[i][eval_joints], bbox, hm_shape=hm_size, norm_type=norm_type)
            pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
            pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
        preds_img = torch.cat(pose_coords)
        preds_scores = torch.cat(pose_scores)
        boxes, scores, ids, preds_img, preds_scores, pick_ids = pose_nms(boxes, scores, ids, preds_img, preds_scores, self.pose_model_args.min_box_area, use_heatmap_loss=(self.pose_model_cfg.DATA_PRESET.get('LOSS_TYPE', 'MSELoss') == 'MSELoss'))

        result = []
        for k in range(len(scores)):
            result.append(
                {
                    'keypoints':preds_img[k],
                    'kp_score':preds_scores[k],
                    'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                    'idx':ids[k],
                    'box':[boxes[k][0], boxes[k][1], boxes[k][2]-boxes[k][0],boxes[k][3]-boxes[k][1]]
                }
            )

        return result

    # run AlphaPose to get the 2D pose
    def get_2d_pose(self, image, boxes) -> dict[str,dict]:
        # process the frame (cropping around the boxes)
        (inps, scores, ids, cropped_boxes) = self.__crop_around_boxes_2d__(image, boxes)
        # Pose Estimation
        with torch.no_grad():
            inps = inps.to(self.pose_model_args.device)
            datalen = inps.size(0)
            leftover = 0
            if datalen % self.pose_model_args.posebatch:
                leftover = 1
            num_batches = datalen // self.pose_model_args.posebatch + leftover
            hm = []
            for j in range(num_batches):
                inps_j = inps[j * self.pose_model_args.posebatch:min((j + 1) * self.pose_model_args.posebatch, datalen)]
                hm_j = self.pose_model(inps_j)
                hm.append(hm_j)
        return self.__heatmap_to_result_dict_2d__((boxes, scores, ids, torch.cat(hm).cpu(), cropped_boxes, image))
