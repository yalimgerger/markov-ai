package com.markovai.server.ai;

public class Patch4x4FeedbackConfig {
    public boolean enabled = false;
    public boolean learningEnabled = false;
    public double adjScale = 0.10;
    public double eta = 0.003;
    public double marginTarget = 0.02;
    public boolean updateOnlyIfIncorrect = true;
    public boolean useMarginGating = true;
    public double maxAdjAbs = 5.0;
    public boolean applyDecayEachEpoch = false;
    public double decayRate = 1.0e-4;

    public boolean frequencyScalingEnabled = false;
    public String frequencyScalingMode = "GLOBAL_SQRT";
    public double minUpdateScale = 0.05;
    public double maxUpdateScale = 1.0;
    public int applyDecayEveryNUpdates = 0;

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
            double decayRate,
            boolean frequencyScalingEnabled,
            String frequencyScalingMode,
            double minUpdateScale,
            double maxUpdateScale,
            int applyDecayEveryNUpdates) {
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
        this.frequencyScalingEnabled = frequencyScalingEnabled;
        this.frequencyScalingMode = frequencyScalingMode;
        this.minUpdateScale = minUpdateScale;
        this.maxUpdateScale = maxUpdateScale;
        this.applyDecayEveryNUpdates = applyDecayEveryNUpdates;
    }

    public static Patch4x4FeedbackConfig disabled() {
        return new Patch4x4FeedbackConfig(false, false, 0.10, 0.003, 0.02, true, true, 5.0, false, 1.0e-4, false,
                "GLOBAL_SQRT", 0.05, 1.0, 0);
    }

    public Patch4x4FeedbackConfig copy() {
        return new Patch4x4FeedbackConfig(
                this.enabled,
                this.learningEnabled,
                this.adjScale,
                this.eta,
                this.marginTarget,
                this.updateOnlyIfIncorrect,
                this.useMarginGating,
                this.maxAdjAbs,
                this.applyDecayEachEpoch,
                this.decayRate,
                this.frequencyScalingEnabled,
                this.frequencyScalingMode,
                this.minUpdateScale,
                this.maxUpdateScale,
                this.applyDecayEveryNUpdates);
    }
}
