# SPDX-License-Identifier: GPL-3.0-only
# From ssitu/ComfyUI_UltimateSDUpscale 1.7.2 modules/shared.py (GPL-3.0); modified 2026-10-10 for ComfyUI-WeightFerry: moved to usdu/a1111/

class Options:
    img2img_background_color = "#ffffff"  # Set to white for now


class State:
    interrupted = False

    def begin(self):
        pass

    def end(self):
        pass


opts = Options()
state = State()

# Will only ever hold 1 upscaler
sd_upscalers = [None]
# The upscaler usable by ComfyUI nodes
actual_upscaler = None

# Batch of images to upscale
batch = []
batch_as_tensor = None
