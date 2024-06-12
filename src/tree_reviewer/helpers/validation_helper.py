
class ValidationHelper:
    @staticmethod
    def validate_tree_image(tree_image):
        error_tags = []
        if tree_image.hypothesis_score < 0.8:
            error_tags.append("Low Hypothesis score")
        if tree_image.classification_score < 0.4:
            error_tags.append("Low Classification score")
        if tree_image.tree_mask.crown_mask is None:
            error_tags.append("Bad Trunk and Crown")
        return error_tags

    @staticmethod
    def validate_segmented_card(segmented_card):
        error_tags = []
        if segmented_card is None:
            error_tags.append("No card detected")
        return error_tags