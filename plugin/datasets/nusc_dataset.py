from.base_dataset import BaseMapDataset
from .map_utils.nuscmap_extractor import NuscMapExtractor
from mmdet.datasets import DATASETS
import numpy as np
from .visualize.renderer import Renderer
import mmcv
from time import time
from pyquaternion import Quaternion
import math
import pyquaternion
from shapely.geometry import LineString
from os import path as osp


@DATASETS.register_module()
class NuscDataset(BaseMapDataset):
    """NuScenes map dataset class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """
    
    def __init__(self, data_root, map_ann_file=None, **kwargs):
        super().__init__(**kwargs)
        self.map_extractor = NuscMapExtractor(data_root, self.roi_size)
        self.renderer = Renderer(self.cat2id, self.roi_size, 'nusc')
        self.map_annos = self.load_map_annotations(map_ann_file)

    def load_map_annotations(self, map_ann_file):
        if map_ann_file is None:
            return None
        if (not osp.exists(map_ann_file)):
            print('Start to convert gt map format...')
            dataset_length = len(self)
            prog_bar = mmcv.ProgressBar(dataset_length)
            map_annos = []
            for sample_id in range(dataset_length):
                sample = self.samples[sample_id]
                location = sample['location']
                map_geoms = self.map_extractor.get_map_geom(location, sample['e2g_translation'], 
                    sample['e2g_rotation'])
                map_anno = self.geom2anno(map_geoms)
                map_annos.append(map_anno)
                prog_bar.update()
            mmcv.dump(map_annos, map_ann_file)
            print('\n Map annos writes to', map_ann_file)

        map_annos = mmcv.load(map_ann_file)
        return map_annos

    def geom2anno(self, map_geoms):
        vectors = {}
        MAP_CLASSES = (
            'ped_crossing',
            'divider',
            'boundary',
        )
        for cls, geom_list in map_geoms.items():
            if cls in MAP_CLASSES:
                label = MAP_CLASSES.index(cls)
                vectors[label] = []
                for geom in geom_list:
                    line = np.array(geom.coords)
                    vectors[label].append(line)
        return vectors

    def anno2geom(self, annos):
        map_geoms = {}
        for label, anno_list in annos.items():
            map_geoms[label] = []
            for anno in anno_list:
                geom = LineString(anno)
                map_geoms[label].append(geom)
        return map_geoms

    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        
        start_time = time()
        ann = mmcv.load(ann_file)
        samples = ann[::self.interval]
        samples = list(sorted(samples, key=lambda e: e["timestamp"]))
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples

    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        location = sample['location']

        if self.map_annos is not None:
            map_label2geom = self.anno2geom(self.map_annos[idx])
        else:
            map_geoms = self.map_extractor.get_map_geom(location, sample['e2g_translation'], 
                    sample['e2g_rotation'])
            map_label2geom = {}
            for k, v in map_geoms.items():
                if k in self.cat2id.keys():
                    map_label2geom[self.cat2id[k]] = v

        ego2img_rts = []
        for c in sample['cams'].values():
            extrinsic, intrinsic = np.array(
                c['extrinsics']), np.array(c['intrinsics'])
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)

        # if sample['sample_idx'] == 0:
        #     is_first_frame = True
        # else:
        #     is_first_frame = self.flag[sample['sample_idx']] > self.flag[sample['sample_idx'] - 1]
        ego2global = np.eye(4)
        ego2global[:3, :3] = pyquaternion.Quaternion(
            sample["e2g_rotation"]
        ).rotation_matrix
        ego2global[:3, 3] = np.array(sample["e2g_translation"])
        input_dict = {
            'location': location,
            'token': sample['token'],
            'timestamp': sample['timestamp'],
            'img_filenames': [c['img_fpath'] for c in sample['cams'].values()],
            # intrinsics are 3x3 Ks
            'cam_intrinsics': [c['intrinsics'] for c in sample['cams'].values()],
            # extrinsics are 4x4 tranform matrix, **ego2cam**
            'cam_extrinsics': [c['extrinsics'] for c in sample['cams'].values()],
            'ego2img': ego2img_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': sample['e2g_translation'], 
            'ego2global_rotation': Quaternion(sample['e2g_rotation']).rotation_matrix.tolist(),
            # 'is_first_frame': is_first_frame, # deprecated
            'sample_idx': sample['sample_idx'],
            'scene_name': sample['scene_name'],
            'ego2global': ego2global,
            # 'group_idx': self.flag[sample['sample_idx']]
        }

        return input_dict