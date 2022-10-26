# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import torch
from torch import nn

from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .fsod_roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

from detectron2.modeling.poolers import ROIPooler
import torch.nn.functional as F

from .fsod_fast_rcnn import FsodFastRCNNOutputs

import os

#import matplotlib.pyplot as plt
import pandas as pd

from detectron2.data.catalog import MetadataCatalog
import detectron2.data.detection_utils as utils
import pickle
import sys

__all__ = ["FsodRCNN_Paisley"]


@META_ARCH_REGISTRY.register()
class FsodRCNN_Paisley(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES

        self.support_way = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot = cfg.INPUT.FS.SUPPORT_SHOT
        
        self.use_contrastive_loss = cfg.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.USE_SUPCONLOSS
        self.contrast_loss_wt = cfg.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.LOSS_WEIGHT
        self.contrast_iou_threshold = cfg.MODEL.ROI_HEADS.CONTRASTIVE_BRANCH.IOU_THRESHOLD
        
        self.logger = logging.getLogger(__name__)

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            self.init_model()
            return self.inference(batched_inputs)
        
        images, support_images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            for x in batched_inputs:
                x['instances'].set('gt_classes', torch.full_like(x['instances'].get('gt_classes'), 0))
            
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        features = self.backbone(images.tensor) # res4  # 2X1024x38x56  # batchsize =2 

        # support branches
        support_bboxes_ls = []
        for item in batched_inputs:
            bboxes = item['support_bboxes']
            for box in bboxes:
                box = Boxes(box[np.newaxis, :])
                support_bboxes_ls.append(box.to(self.device))   # list of 8*20=160 support boxes
        
        B, N, C, H, W = support_images.tensor.shape
        assert N == self.support_way * self.support_shot

        support_images = support_images.tensor.reshape(B*N, C, H, W)  # 18x3x320x320
        support_features = self.backbone(support_images)    # res4 [18, 1024, 20, 20]  # 18 with batchsize of 2 images
        
        # support feature roi pooling
        feature_pooled = self.roi_heads.roi_pooling(support_features, support_bboxes_ls) # [40, 1024, 14, 14]
        # support box feature after res5 layer 
        support_box_features = self.roi_heads._shared_roi_transform([support_features[f] for f in self.in_features], support_bboxes_ls) #[40, 2048, 7, 7]
        #assert self.support_way == 2 # now only 2 way support
        
        # Additionally compute contrastive feature for support boxes
        if self.use_contrastive_loss: 
            support_box_contrast_features = self.roi_heads.encoder(support_box_features.mean(dim=[-1,-2]))
        
        
        
        detector_loss_cls = []
        detector_loss_box_reg = []
        rpn_loss_rpn_cls = []
        rpn_loss_rpn_loc = []
        pos_proposal_confeats = [] # positive  proposal contrast features
        pos_proposal_gt_classes = []  # accumualte positive proposal groudtruth classes
        pos_proposal_ious = []  # Compute the ious of the proposal 
        
        for i in range(B): # batch
            # query
            query_gt_instances = [gt_instances[i]] # one query gt instances
            query_images = ImageList.from_tensors([images[i]]) # one query image

            query_feature_res4 = features['res4'][i].unsqueeze(0) # one query feature for attention rpn
            query_features = {'res4': query_feature_res4} # one query feature for rcnn

            # positive support branch ##################################
            pos_begin = i * self.support_shot * self.support_way
            pos_end = pos_begin + self.support_shot
            pos_support_features = feature_pooled[pos_begin:pos_end].mean(0, True) # [10, 1024, 14, 14] --mean -->   pos support features from res4, average all supports, for rcnn
            pos_support_features_pool = pos_support_features.mean(dim=[2, 3], keepdim=True) # [1, 1024, 1, 1] average pooling support feature for attention rpn
            pos_correlation = F.conv2d(query_feature_res4, pos_support_features_pool.permute(1,0,2,3), groups=1024) # attention map #[1, 1024, 38, 50]

            pos_features = {'res4': pos_correlation} # attention map for attention rpn
            pos_support_box_features = support_box_features[pos_begin:pos_end].mean(0, True)  #[1, 2048, 7, 7]
            pos_proposals, pos_anchors, pos_pred_objectness_logits, pos_gt_labels, pos_pred_anchor_deltas, pos_gt_boxes = self.proposal_generator(query_images, pos_features, query_gt_instances) # attention rpn
            if self.use_contrastive_loss:            
                pos_pred_class_logits, pos_pred_proposal_deltas, pos_detector_proposals, pos_proposals_contrast_features = self.roi_heads(query_images, query_features, pos_support_box_features, pos_proposals, query_gt_instances) # pos rcnn
                pos_proposal_confeats.append(pos_proposals_contrast_features)
                pos_proposal_gt_classes.append(pos_detector_proposals[0].gt_classes)
                pos_proposal_ious.append(pos_detector_proposals[0].iou)
            else:
                pos_pred_class_logits, pos_pred_proposal_deltas, pos_detector_proposals = self.roi_heads(query_images, query_features, pos_support_box_features, pos_proposals, query_gt_instances) # pos rcnn

            # # negative support branch ##################################
            # neg_begin = pos_end 
            # neg_end = neg_begin + self.support_shot 

            # neg_support_features = feature_pooled[neg_begin:neg_end].mean(0, True)
            # neg_support_features_pool = neg_support_features.mean(dim=[2, 3], keepdim=True)
            # neg_correlation = F.conv2d(query_feature_res4, neg_support_features_pool.permute(1,0,2,3), groups=1024)

            # neg_features = {'res4': neg_correlation}

            # neg_support_box_features = support_box_features[neg_begin:neg_end].mean(0, True)
            # neg_proposals, neg_anchors, neg_pred_objectness_logits, neg_gt_labels, neg_pred_anchor_deltas, neg_gt_boxes = self.proposal_generator(query_images, neg_features, query_gt_instances)
            # neg_pred_class_logits, neg_pred_proposal_deltas, neg_detector_proposals = self.roi_heads(query_images, query_features, neg_support_box_features, neg_proposals, query_gt_instances)

            # rpn loss
            #outputs_images = ImageList.from_tensors([images[i], images[i]])

            # outputs_pred_objectness_logits = [torch.cat(pos_pred_objectness_logits + neg_pred_objectness_logits, dim=0)]
            # outputs_pred_anchor_deltas = [torch.cat(pos_pred_anchor_deltas + neg_pred_anchor_deltas, dim=0)]
            
            # outputs_anchors = pos_anchors # + neg_anchors

            # # convert 1 in neg_gt_labels to 0
            # for item in neg_gt_labels:
            #     item[item == 1] = 0

            # outputs_gt_boxes = pos_gt_boxes + neg_gt_boxes #[None]
            # outputs_gt_labels = pos_gt_labels + neg_gt_labels
            
            if self.training:
                proposal_losses = self.proposal_generator.losses(
                    pos_anchors, pos_pred_objectness_logits, pos_gt_labels, pos_pred_anchor_deltas, pos_gt_boxes)
                proposal_losses = {k: v * self.proposal_generator.loss_weight for k, v in proposal_losses.items()}
            else:
                proposal_losses = {}

            # detector loss
            # detector_pred_class_logits = torch.cat([pos_pred_class_logits, neg_pred_class_logits], dim=0)
            # detector_pred_proposal_deltas = torch.cat([pos_pred_proposal_deltas, neg_pred_proposal_deltas], dim=0)
            # for item in neg_detector_proposals:
            #     item.gt_classes = torch.full_like(item.gt_classes, 1)
            
            #detector_proposals = pos_detector_proposals + neg_detector_proposals
            # detector_proposals = [Instances.cat(pos_detector_proposals + neg_detector_proposals)]
            if self.training:
                predictions = pos_pred_class_logits, pos_pred_proposal_deltas
                detector_losses = self.roi_heads.box_predictor.losses(predictions, pos_detector_proposals)

            rpn_loss_rpn_cls.append(proposal_losses['loss_rpn_cls'])
            rpn_loss_rpn_loc.append(proposal_losses['loss_rpn_loc'])
            detector_loss_cls.append(detector_losses['loss_cls'])
            detector_loss_box_reg.append(detector_losses['loss_box_reg'])
        
        proposal_losses = {}
        detector_losses = {}

        proposal_losses['loss_rpn_cls'] = torch.stack(rpn_loss_rpn_cls).mean()
        proposal_losses['loss_rpn_loc'] = torch.stack(rpn_loss_rpn_loc).mean()
        detector_losses['loss_cls'] = torch.stack(detector_loss_cls).mean() 
        detector_losses['loss_box_reg'] = torch.stack(detector_loss_box_reg).mean()

        # Compute the contrastive loss
        pos_roi_feats = torch.cat(pos_proposal_confeats, dim=0)
        pos_roi_labels = torch.cat(pos_proposal_gt_classes, dim=0)
        pos_roi_ious = torch.cat(pos_proposal_ious, dim=0)
        
        #index  of all background proposals and foreground proposal that  has iou greater than threshold
        # background acts as negative samples and foreground as postives  for contrastive learnin. 
        keep =  (pos_roi_ious >= self.contrast_iou_threshold) + (pos_roi_labels==1)
        pos_roi_feats = pos_roi_feats[keep,:]
        pos_roi_labels = pos_roi_labels[keep]
        
        
        support_labels = pos_roi_labels.new_full((support_box_contrast_features.shape[0],1), 1).squeeze() #same dtype and device as pos_roi_label, labels are all 0 (all paisley as support)
        all_contrast_feats = torch.cat((pos_roi_feats, support_box_contrast_features), dim=0)
        all_contrast_labels = torch.cat((pos_roi_labels, support_labels), dim=0)
        contrast_loss = self.roi_heads.contrastive_criterion(all_contrast_feats, all_contrast_labels)
        
        detector_losses['loss_contrastive'] = contrast_loss * self.contrast_loss_wt
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
    
    @torch.no_grad()
    def init_model(self):
        self.support_on = True #False

        support_dir = './support_dir'
        if not os.path.exists(support_dir):
            os.makedirs(support_dir)

        support_file_name = os.path.join(support_dir, 'support_feature_paisley.pkl')
        if not os.path.exists(support_file_name):
            support_path = './datasets/paisley/10_shot_support_df.pkl'
            support_df = pd.read_pickle(support_path)

            metadata = MetadataCatalog.get('paisley_test')
            # unmap the category mapping ids for COCO
            #reverse_id_mapper = lambda dataset_id: metadata.thing_dataset_id_to_contiguous_id[dataset_id]  # noqa
            #support_df['category_id'] = support_df['category_id'].map(reverse_id_mapper)

            support_dict = {'res4_avg': {}, 'res5_avg': {}}
            for cls in support_df['category_id'].unique():
                support_cls_df = support_df.loc[support_df['category_id'] == cls, :].reset_index()
                support_data_all = []
                support_box_all = []

                for index, support_img_df in support_cls_df.iterrows():
                    img_path =  support_img_df['file_path']
                    support_data = utils.read_image(img_path, format='BGR')
                    support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                    support_data_all.append(support_data)

                    support_box = support_img_df['support_box']
                    support_box_all.append(Boxes([support_box]).to(self.device))

                # support images
                support_images = [x.to(self.device) for x in support_data_all]
                support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
                support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
                                
                N = len(support_images)
                ind_list = np.array_split(range(N), int(N/10))

                res4_pooled_all = []
                res5_feature_all = [] 
                for ind in ind_list :
                    support_features_temp = self.backbone(support_images.tensor[ind,:])
                    support_box_temp = [support_box_all[x] for x in ind]
                    res4_pooled = self.roi_heads.roi_pooling(support_features_temp, support_box_temp)
                    res4_pooled = res4_pooled.detach().cpu()
                    res4_pooled_all.append(res4_pooled)
                    
                    res5_feature_tmp = self.roi_heads._shared_roi_transform([support_features_temp[f] for f in self.in_features], support_box_temp)
                    res5_feature_all.append(res5_feature_tmp.detach().cuda())
                    
                res4_pooled = torch.cat(res4_pooled_all, dim=0)
                res4_avg = res4_pooled.mean(0, True)
                res4_avg = res4_avg.mean(dim=[2,3], keepdim=True)

                res5_feature = torch.cat(res5_feature_all, dim=0)
                res5_avg = res5_feature.mean(0, True)
                 
                support_dict['res4_avg'][cls] = res4_avg.data
                support_dict['res5_avg'][cls] = res5_avg.data
                
                
                del res4_avg
                del res4_pooled
                del support_features_temp
                del res5_feature
                del res5_avg

            with open(support_file_name, 'wb') as f:
               pickle.dump(support_dict, f)
            self.logger.info("=========== Offline support features are generated. ===========")
            self.logger.info("============ Few-shot object detetion will start. =============")
            sys.exit(0)
            
        else:
            with open(support_file_name, "rb") as hFile:
                self.support_dict  = pickle.load(hFile, encoding="latin1")
            for res_key, res_dict in self.support_dict.items():
                for cls_key, feature in res_dict.items():
                    self.support_dict[res_key][cls_key] = feature.cuda()

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        B, _, _, _ = features['res4'].shape
        assert B == 1 # only support 1 query image in test
        assert len(images) == 1
        support_proposals_dict = {}
        support_box_features_dict = {}
        proposal_num_dict = {}
 
        for cls_id, res4_avg in self.support_dict['res4_avg'].items():
            query_images = ImageList.from_tensors([images[0]]) # one query image

            query_features_res4 = features['res4'] # one query feature for attention rpn
            query_features = {'res4': query_features_res4} # one query feature for rcnn

            # support branch ##################################
            support_box_features = self.support_dict['res5_avg'][cls_id]

            correlation = F.conv2d(query_features_res4, res4_avg.permute(1,0,2,3), groups=1024) # attention map

            support_correlation = {'res4': correlation} # attention map for attention rpn

            proposals, _ = self.proposal_generator(query_images, support_correlation, None)
            support_proposals_dict[cls_id] = proposals
            support_box_features_dict[cls_id] = support_box_features

            if cls_id not in proposal_num_dict.keys():
                proposal_num_dict[cls_id] = []
            proposal_num_dict[cls_id].append(len(proposals[0]))

            del support_box_features
            del correlation
            del res4_avg
            del query_features_res4

        results, _ = self.roi_heads.eval_with_support(query_images, query_features, support_proposals_dict, support_box_features_dict)
        
        if do_postprocess:
            return FsodRCNN_Paisley._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        if self.training:
            # support images
            support_images = [x['support_images'].to(self.device) for x in batched_inputs]
            support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
            support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)

            return images, support_images
        else:
            return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results
