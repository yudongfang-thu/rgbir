"""Reuse the original OEv1 C0 math; enable its existing sanity observer."""


def make_type(original):
    class ConfidenceCriterion(original):
        def __call__(self, prediction, batch):
            # build_trainer(max_steps=None) otherwise disables old score-gradient
            # checks. New canaries observe every batch; full FT observes first.
            self.sanity = bool(self.cfg.get('canary_execution', False)) or self.calls == 0
            return super().__call__(prediction, batch)
    return ConfidenceCriterion
