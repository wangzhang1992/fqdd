import torch.nn.functional as F


def align_features(feature1, feature2):
    len1 = feature1.shape[0]
    len2 = feature2.shape[0]

    if len1 > len2:
        feature2 = F.interpolate(
            feature2.unsqueeze(0).unsqueeze(0),
            size=(len1, feature2.shape[1]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)
    elif len2 > len1:
        feature1 = F.interpolate(
            feature1.unsqueeze(0).unsqueeze(0),
            size=(len2, feature1.shape[1]),
            mode='bilinear',
            align_corners=False
        ).squeeze(0).squeeze(0)

    return feature1, feature2
