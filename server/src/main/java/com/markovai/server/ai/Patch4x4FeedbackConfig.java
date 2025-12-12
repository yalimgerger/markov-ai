package com.markovai.server.ai;

public class Patch4x4FeedbackConfig {
    public boolean enabled;
    public boolean learningEnabled;
    public double adjScale;
    public double eta;
    public double marginTarget;
    public boolean updateOnlyIfIncorrect;
    public boolean useMarginGating;
    public double maxAdjAbs;
    public boolean applyDecayEachEpoch;
    public double decayRate;

    public Patch4x4FeedbackConfig() {
    }

    public Patch4x4FeedbackConfig(
            boolean enabled,
            boolean learningEnabled,
            double adjScale,
            double eta,
            double marginTarget,
            boolean updateOnlyIfIncorrect,
            boolean useMarginGating,
            double maxAdjAbs,
            boolean applyDecayEachEpoch,
            double decayRate) {
        this.enabled = enabled;
        this.learningEnabled = learningEnabled;
        this.adjScale = adjScale;
        this.eta = eta;
        this.marginTarget = marginTarget;
        this.updateOnlyIfIncorrect = updateOnlyIfIncorrect;
        this.useMarginGating = useMarginGating;
        this.maxAdjAbs = maxAdjAbs;
        this.applyDecayEachEpoch = applyDecayEachEpoch;
        this.decayRate = decayRate;
    }

    public static Patch4x4FeedbackConfig disabled() {
        return new Patch4x4FeedbackConfig(false, false, 0.10, 0.003, 0.02, true, true, 5.0, false, 1.0e-4);
    }
}
