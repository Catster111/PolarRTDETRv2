import re

# Read the criterion file
with open('src/zoo/rtdetr/polar_face_criterion.py', 'r') as f:
    content = f.read()

# Replace the landmark loss computation with proper normalization
old_code = '''        # Normalize landmarks to [0, 1] based on image size
        # Assuming landmarks are already normalized in the dataset
        loss_landmarks = F.smooth_l1_loss(src_landmarks, target_landmarks, reduction='none')'''

new_code = '''        # Normalize landmarks to [0, 1] based on image size (landmarks are in pixel coordinates)
        image_sizes = []
        for t, (_, i) in zip(targets, indices):
            for _ in range(len(i)):
                image_sizes.append(t["orig_size"])
        
        if image_sizes:
            image_sizes = torch.stack(image_sizes, dim=0)
            h, w = image_sizes[:, 0:1], image_sizes[:, 1:2]  # [N, 1]
            
            # Normalize target landmarks to [0, 1]
            target_landmarks_norm = target_landmarks.clone()
            target_landmarks_norm[:, 0::2] = target_landmarks_norm[:, 0::2] / w  # x coordinates
            target_landmarks_norm[:, 1::2] = target_landmarks_norm[:, 1::2] / h  # y coordinates
            
            loss_landmarks = F.smooth_l1_loss(src_landmarks, target_landmarks_norm, reduction='none')
        else:
            loss_landmarks = F.smooth_l1_loss(src_landmarks, target_landmarks, reduction='none')'''

content = content.replace(old_code, new_code)

# Write back
with open('src/zoo/rtdetr/polar_face_criterion.py', 'w') as f:
    f.write(content)
