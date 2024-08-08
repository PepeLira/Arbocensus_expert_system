from tree_reviewer.config import get_env

class ValidationHelper:
    @staticmethod
    def validate_tree_image(tree_image):
        error_tags = []
        if tree_image.hypothesis_score < float(get_env("HYPOTHESIS_THRESHOLD")):
            error_tags.append("Low Hypothesis score")
        if tree_image.classification_score < float(get_env("TREE_VALIDATION_SCORE")):
            error_tags.append("Low Classification score")
        if tree_image.tree_mask.crown_mask is None:
            error_tags.append("Bad Trunk and Crown")
        if tree_image.species_confidence < float(get_env("SPECIES_CONFIDENCE")):
            error_tags.append("Low species confidence")
        return error_tags

    @staticmethod
    def validate_segmented_card(segmented_card, pixel_diameter):
        error_tags = []
        if len(segmented_card.array) < 5:
            error_tags.append("No card detected")
        elif segmented_card.binary_mask is None or segmented_card.mm_per_pixel is None:
            error_tags.append("No card detected")
        elif segmented_card.l_side > pixel_diameter + 30:
            error_tags.append("No card detected")
        return error_tags
    
    @staticmethod
    def validate_metrics(tree_metrics):
        error_tags = []
        if tree_metrics.dap < float(get_env("LOW_DAP")):
            error_tags.append("Bad tree DAP")
        if tree_metrics.dap > float(get_env("HIGH_DAP")):
            error_tags.append("Bad tree DAP")
        if tree_metrics.tree_height < float(get_env("LOW_HEIGHT")):
            error_tags.append("Bad tree height")
        if tree_metrics.tree_height > float(get_env("MAX_HEIGHT")):
            error_tags.append("Bad tree height")
        if tree_metrics.principal_branches_h < float(get_env("LOW_BRANCHES")):
            error_tags.append("Bad branches height")
        if tree_metrics.principal_branches_h > float(get_env("MAX_BRANCHES")):
            error_tags.append("Bad branches height")
        return error_tags