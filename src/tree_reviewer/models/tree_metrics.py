from .helpers.metrics_helper import height_from_tree_mask, principal_branches_height
from .helpers.metrics_helper import dap_from_tree_mask

class TreeMetrics:
    def __init__(self, tree_image, decimal_digits=2):
        self.tree_image = tree_image
        self.error_tags = []
        self.precision = decimal_digits
        self.valid = True

        self.mm_per_pixel = None
        self.tree_height = None
        self.principal_branches_h = None
        self.dap = None

        self.select_tree_image_data()

        self.tree_image_data = None
        self.metrics = None
    
    def select_tree_image_data(self):
        self.file = self.tree_image.file
        self.shape = self.tree_image.shape
        if self.tree_image.classification_score is not None and self.tree_image.hypothesis_score is not None:
            self.classification_score = round(self.tree_image.classification_score, self.precision)
            self.hypothesis_score = round(self.tree_image.hypothesis_score, self.precision)
        self.trunk_xyxy = self.tree_image.get_trunk_xyxy_bbox()
        self.mask_xyxy = self.tree_image.get_mask_xyxy_bbox()
    
    def define_metrics(self, mm_per_pixel):
        tree_mask = self.tree_image.tree_mask.binary_mask
        trunk_mask = self.tree_image.tree_mask.trunk_mask
        self.mm_per_pixel = mm_per_pixel

        self.tree_height = height_from_tree_mask(tree_mask, mm_per_pixel)
        self.principal_branches_h = principal_branches_height(trunk_mask, mm_per_pixel)
        self.dap = dap_from_tree_mask(tree_mask, mm_per_pixel, self.tree_height)

        self.tree_image.define_dap(self.dap)
        self.tree_image.define_height(self.tree_height)
        self.tree_image.define_principal_branches_height(self.principal_branches_h)

        return self.tree_image
    
    def group_metrics(self):
        self.replace_null_data()
        self.round_metrics()
        self.round_tree_image_data()

        self.tree_image_data = {'shape': self.shape,
                        'classification_score': self.classification_score,
                        'hypothesis_score': self.hypothesis_score,
                        'trunk_xyxy': self.trunk_xyxy,
                        'mask_xyxy': self.mask_xyxy}
        
        self.metrics = {'DAP': self.dap,
                        'height': self.tree_height,
                        'principal_branches_height': self.principal_branches_h}
        
    def round_metrics(self):
        self.dap = float(round(self.dap, self.precision))
        self.tree_height = float(round(self.tree_height, self.precision))
        self.principal_branches_h = float(round(self.principal_branches_h, self.precision))

    def round_tree_image_data(self):
        self.classification_score = round(float(self.classification_score), self.precision)
        self.hypothesis_score = round(float(self.hypothesis_score), self.precision)
        self.trunk_xyxy = [float(round(i, self.precision)) for i in self.trunk_xyxy]
        self.mask_xyxy = [float(round(i, self.precision)) for i in self.mask_xyxy]

    def replace_null_data(self):
        if self.dap is None:
            self.dap = 0
        if self.tree_height is None:
            self.tree_height = 0
        if self.principal_branches_h is None:
            self.principal_branches_h = 0
        if self.classification_score is None:
            self.classification_score = 0
        if self.hypothesis_score is None:
            self.hypothesis_score = 0
        if self.trunk_xyxy is None:
            self.trunk_xyxy = [0, 0, 0, 0]
        if self.mask_xyxy is None:
            self.mask_xyxy = [0, 0, 0, 0]

    def get_data(self):
        if len(self.error_tags) > 2 or "No card detected" in self.error_tags:
            self.valid = False
        data = {'id': self.file.split('.')[-2],
                'valid': self.valid,
                'image_data': self.tree_image_data, 
                'metrics': self.metrics,
                'errors': self.error_tags}
        return data
    
    def set_blank_data(self):
        self.valid = False
        self.dap = 0
        self.tree_height = 0
        self.principal_branches_h = 0
        self.classification_score = 0
        self.hypothesis_score = 0
        self.trunk_xyxy = [0, 0, 0, 0]
        self.mask_xyxy = [0, 0, 0, 0]
        self.error_tags = ["No card detected", "No tree detected"]
        self.group_metrics()
    
    def assign_error_tags(self, error_tags):
        self.error_tags += error_tags