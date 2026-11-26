# First-frame visual review

**Review status: visually confirmed for this frame.** The user reports manually inspecting a screenshot from the first frame of `videos/8964295-uhd_3840_2160_25fps.mp4`. The screenshot itself was not supplied as part of this documentation update.

- Frame 0, timestamp 0.0 seconds: the user reports that the right-side worker visibly wears a helmet and lacks a safety vest. In the saved inference this is person 2, with helmet `PRESENT` and vest `MISSING`; the saved rules output contains a corresponding vest incident candidate supported by a `NoVest` detection at confidence 0.6171.
- In the same frame, the user reports that the left-side worker visibly wears both helmet and vest. The saved inference identifies this person as person 1, with both statuses `PRESENT`.

This is a human-reported visual confirmation of the example in this specific frame. It does not establish overall model or association accuracy and is not a confirmed real-world safety violation. Person IDs are frame-local. The original `incidents.json` retains its initial `unverified_model_output` status as a historical inference artifact; this separate document records the subsequent review.
