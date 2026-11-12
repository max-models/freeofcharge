"""The scope-profiler tag of freeofcharge's profiling regions.

The regions are opt-in: `freeofcharge` registers the tag on import, so they record
nothing unless the application enables it, e.g. with
``scope_profiler.setup(enable_tags=["freeofcharge"])``.
"""

#: Tags of every freeofcharge region; the first is the opt-in tag.
PROFILE_TAGS = ("freeofcharge",)
