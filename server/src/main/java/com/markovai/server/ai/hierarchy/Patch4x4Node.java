package com.markovai.server.ai.hierarchy;

import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.DigitMarkovModel;
import com.markovai.server.ai.DigitPatch4x4UnigramModel;
import com.markovai.server.ai.Patch4x4FeedbackConfig;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class Patch4x4Node implements DigitFactorNode {
    private static final Logger logger = LoggerFactory.getLogger(Patch4x4Node.class);

    private final String id;
    private final DigitPatch4x4UnigramModel model;
    private final double smoothingLambda;
    private Patch4x4FeedbackConfig feedbackCfg;

    public Patch4x4FeedbackConfig getFeedbackConfig() {
        return feedbackCfg;
    }

    // Adjustment table: [digit][symbol]
    // 0..65535 symbols. 10 digits.
    private final double[][] adj = new double[10][65536];
    private final long[] globalSymbolCount = new long[65536];
    private long updatesCounter = 0;

    public Patch4x4Node(String id, DigitPatch4x4UnigramModel model, double smoothingLambda,
            Patch4x4FeedbackConfig feedbackCfg) {
        this.id = id;
        this.model = model;
        this.smoothingLambda = smoothingLambda;
        this.feedbackCfg = feedbackCfg;
        if (feedbackCfg == null) {
            throw new IllegalArgumentException("feedbackCfg cannot be null");
        }
    }

    @Override
    public String getId() {
        return id;
    }

    @Override
    public List<DigitFactorNode> getChildren() {
        return Collections.emptyList();
    }

    @Override
    public NodeResult computeForImage(DigitImage img, Map<String, NodeResult> childResults) {
        int[] symbols = extractPatchSymbols(img);
        int nSteps = symbols.length; // Should be 49

        double[] smoothedSum = new double[10];

        for (int d = 0; d < 10; d++) {
            double prevLp = 0.0;
            boolean first = true;
            double sum = 0.0;

            for (int symbol : symbols) {
                double baseLp = model.logProbForSymbol(d, symbol);

                // Apply feedback adjustment if enabled
                if (feedbackCfg.enabled) {
                    baseLp += feedbackCfg.adjScale * adj[d][symbol];
                }

                double lp = baseLp;

                if (first) {
                    sum += lp;
                    first = false;
                } else {
                    double diff = lp - prevLp;
                    sum += lp - smoothingLambda * Math.abs(diff);
                }
                prevLp = lp;
            }
            smoothedSum[d] = sum;
        }

        double[] avgLogL = new double[10];
        for (int d = 0; d < 10; d++) {
            avgLogL[d] = smoothedSum[d] / nSteps;
        }

        if (logger.isDebugEnabled()) {
            double minAvg = Double.MAX_VALUE;
            double maxAvg = Double.MIN_VALUE;
            for (double val : avgLogL) {
                if (val < minAvg)
                    minAvg = val;
                if (val > maxAvg)
                    maxAvg = val;
            }
            logger.debug("Patch4x4Node {} [Label {}]: AvgLogL range=[{}, {}]",
                    id, img.label, String.format("%.4f", minAvg), String.format("%.4f", maxAvg));
        }

        return new NodeResult(avgLogL);
    }

    public int[] extractPatchSymbols(DigitImage img) {
        // Binarize
        int[][] binary2D = DigitMarkovModel.binarize(img.pixels, 128);

        int[] symbols = new int[49];
        int idx = 0;
        for (int r = 0; r < 7; r++) {
            for (int c = 0; c < 7; c++) {
                symbols[idx++] = DigitPatch4x4UnigramModel.encodePatch(binary2D, r * 4, c * 4);
            }
        }
        return symbols;
    }

    public void setFeedbackConfig(Patch4x4FeedbackConfig newConfig) {
        if (newConfig == null) {
            throw new IllegalArgumentException("Cannot set null feedback config");
        }
        this.feedbackCfg = newConfig;
    }

    public void applyFeedback(int[] symbols, int trueDigit, int rivalDigit, boolean wasCorrect, double margin) {
        applyFeedback(symbols, trueDigit, rivalDigit, wasCorrect, margin, 1.0);
    }

    public void applyFeedback(int[] symbols, int trueDigit, int rivalDigit, boolean wasCorrect, double margin,
            double updateScale) {
        // Must be enabled overall
        if (!feedbackCfg.enabled)
            return;

        // Must have learning specifically enabled
        if (!feedbackCfg.learningEnabled) {
            // Scoring is enabled but learning is frozen.
            return;
        }

        // If external scale is 0, skip
        if (updateScale <= 0.0) {
            return;
        }

        boolean doUpdate = true;
        if (feedbackCfg.updateOnlyIfIncorrect && wasCorrect) {
            doUpdate = false;
        }

        if (feedbackCfg.useMarginGating) {
            if (margin >= feedbackCfg.marginTarget) {
                doUpdate = false;
            }
        }

        if (!doUpdate)
            return;

        double scale = 1.0;
        if (feedbackCfg.useMarginGating) {
            scale = (feedbackCfg.marginTarget - margin);
            if (scale < 0)
                scale = 0;
            if (scale > 1.0)
                scale = 1.0; // clamp
        }

        // Apply external scale
        scale *= updateScale;

        for (int s : symbols) {
            if (feedbackCfg.learningEnabled) {
                globalSymbolCount[s]++;
            }

            double scaleFreq = 1.0;
            if (feedbackCfg.frequencyScalingEnabled) {
                if ("GLOBAL_LINEAR".equals(feedbackCfg.frequencyScalingMode)) {
                    scaleFreq = 1.0 / (1.0 + globalSymbolCount[s]);
                } else {
                    // Default to GLOBAL_SQRT
                    scaleFreq = 1.0 / Math.sqrt(1.0 + globalSymbolCount[s]);
                }
                if (scaleFreq < feedbackCfg.minUpdateScale)
                    scaleFreq = feedbackCfg.minUpdateScale;
                if (scaleFreq > feedbackCfg.maxUpdateScale)
                    scaleFreq = feedbackCfg.maxUpdateScale;
            }

            double etaEff = feedbackCfg.eta * scale * scaleFreq;

            adj[trueDigit][s] += etaEff;
            adj[rivalDigit][s] -= etaEff;

            double m = feedbackCfg.maxAdjAbs;
            if (adj[trueDigit][s] > m)
                adj[trueDigit][s] = m;
            if (adj[trueDigit][s] < -m)
                adj[trueDigit][s] = -m;
            if (adj[rivalDigit][s] > m)
                adj[rivalDigit][s] = m;
            if (adj[rivalDigit][s] < -m)
                adj[rivalDigit][s] = -m;
        }

        updatesCounter += symbols.length;
        if (feedbackCfg.applyDecayEveryNUpdates > 0
                && updatesCounter % feedbackCfg.applyDecayEveryNUpdates < symbols.length) {
            // Check if we crossed a multiple of N
            // Simplification: if (updatesCounter % N) is small, or just check the modulo.
            // Strict "every N" means we trigger if we just crossed the threshold.
            // Since we add 49 at a time, we might skip the exact 0.
            // Better check: (oldCounter / N) < (newCounter / N)
            long oldVal = updatesCounter - symbols.length;
            if (oldVal / feedbackCfg.applyDecayEveryNUpdates < updatesCounter / feedbackCfg.applyDecayEveryNUpdates) {
                applyDecay(feedbackCfg.decayRate);
            }
        }

        if (logger.isTraceEnabled()) {
            logger.trace("Feedback update: true={}, rival={}, margin={}, scale={}, updated {} symbols",
                    trueDigit, rivalDigit, String.format("%.4f", margin), String.format("%.4f", scale), symbols.length);
        }
    }

    public void applyDecayIfEnabled(boolean isLearningAllowed) {
        if (feedbackCfg.enabled && feedbackCfg.learningEnabled && isLearningAllowed
                && feedbackCfg.applyDecayEachEpoch) {
            applyDecay(feedbackCfg.decayRate);
            logger.info("Applied decay to Patch4x4 adjustments (epoch end)");
        }
    }

    private void applyDecay(double rate) {
        double factor = 1.0 - rate;
        for (int d = 0; d < 10; d++) {
            for (int s = 0; s < 65536; s++) {
                adj[d][s] *= factor;
            }
        }
        if (logger.isDebugEnabled()) {
            logger.debug("Applied decay to Patch4x4 adjustments (factor={})", factor);
        }
    }

    public void resetFeedbackState() {
        for (int d = 0; d < 10; d++) {
            for (int s = 0; s < 65536; s++) {
                adj[d][s] = 0.0;
            }
        }
        for (int s = 0; s < 65536; s++) {
            globalSymbolCount[s] = 0;
        }
        updatesCounter = 0;
        logger.info("Reset Patch4x4 feedback state (adj and counts).");
    }
}
