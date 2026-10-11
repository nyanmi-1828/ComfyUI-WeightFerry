# SPDX-License-Identifier: MIT-0
# Copyright (c) 2026 nyanmi-1828
"""
WeightFerrySamplerChoice: one sampler name / scheduler for several samplers of a workflow
(KSampler, FaceDetailer, DetailerForEach, Ultimate SD Upscale). Outputs the chosen values as combo outputs, which
connect to those nodes' sampler_name / scheduler inputs. The outputs are typed "any": a list-typed output is accepted by
V3 combo inputs (FaceDetailer) but not by legacy nodes whose input type is their own list (DetailerForEach's
scheduler list differs from ComfyUI's -> "return type mismatch").
"""

import comfy.samplers


class _AnyType(str):
    """The legacy "any" type: equal to every type in ComfyUI's link validation."""

    def __ne__(self, other):
        return False


_ANY = _AnyType("*")


class WeightFerrySamplerChoice:
    @classmethod
    def INPUT_TYPES(cls):
        # read at request time: other packs (bleh presets, "real", "tors") add their names after this module loads
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Sampler shared by every stage (base, face detail, inpaint, upscale)"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Scheduler shared by every stage"}),
            }
        }

    RETURN_TYPES = (_ANY, _ANY)
    RETURN_NAMES = ("sampler_name", "scheduler")
    FUNCTION = "choose"
    CATEGORY = "WeightFerry/sampling"
    TITLE = "Sampler Choice (shared sampler / scheduler)"

    def choose(self, sampler_name, scheduler):
        return (sampler_name, scheduler)

