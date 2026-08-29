"""
test_conditions.py

LOCKED fixed-intensity conditions for Sunday's robustness testing.
Decided once, Saturday night, by D + C — do not change without re-syncing
with the whole team, since B/C/D's Sunday work all depends on these
staying fixed.

These map directly to the transform names already defined in
transforms.py's TRANSFORM_REGISTRY, so C's robustness_eval.py can just
loop over FIXED_TEST_CONDITIONS and call apply_transform() for each one.
"""

# The "hardest case" value was deliberately chosen for each transform
# type (per the schedule) rather than a middle value, so the robustness
# table shows how the model holds up under realistic worst-case
# degradation, not just mild conditions.

FIXED_TEST_CONDITIONS = {
    "jpeg_q30": "JPEG Compression, quality=30 (most degraded)",
    "blur_s2.0": "Gaussian Blur, sigma=2.0 (most degraded)",
    "center_crop_80pct": "Center Crop, 80% (per problem statement)",
}

# Optional bonus condition (only build if time allows, per schedule):
# a stacked "social media simulation" combining JPEG + blur, since
# real-world reposted images often undergo multiple transforms at once.
BONUS_CONDITION = {
    "social_media_sim": "JPEG q=30 + Gaussian Blur sigma=2.0 (stacked)",
}

# Plus the always-included baseline and generalization check:
ALWAYS_INCLUDED = {
    "clean": "No transform applied (baseline)",
    "unseen_generator": "WildFake subset — generator not seen in training",
}


def apply_stacked_social_media_sim(img):
    """
    Applies the bonus stacked transform: JPEG q30 then Gaussian blur
    sigma=2.0, simulating a screenshot-and-reshare scenario.
    Requires transforms.py to be importable.
    """
    from transforms import jpeg_compression, gaussian_blur
    img = jpeg_compression(img, quality=30)
    img = gaussian_blur(img, sigma=2.0)
    return img


if __name__ == "__main__":
    print("Locked fixed-intensity test conditions for Sunday:")
    for name, desc in FIXED_TEST_CONDITIONS.items():
        print(f"  - {name}: {desc}")
    print("\nAlways included:")
    for name, desc in ALWAYS_INCLUDED.items():
        print(f"  - {name}: {desc}")
    print("\nBonus (if time allows):")
    for name, desc in BONUS_CONDITION.items():
        print(f"  - {name}: {desc}")