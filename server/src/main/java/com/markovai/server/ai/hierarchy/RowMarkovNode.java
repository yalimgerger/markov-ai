package com.markovai.server.ai.hierarchy;

import com.markovai.server.ai.CachedMarkovChainEvaluator;
import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.DigitMarkovModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.markovai.server.ai.MultiSequenceExtractor;
import com.markovai.server.ai.Patch4x4FeedbackConfig;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class RowMarkovNode implements DigitFactorNode {

    private static final Logger logger = LoggerFactory.getLogger(RowMarkovNode.class);
    private final String id;
    private final CachedMarkovChainEvaluator evaluator;
    private final MultiSequenceExtractor extractor;
    private Patch4x4FeedbackConfig feedbackConfig;

    // Feedback state
    private static final int NUM_DIGITS = 10;
    private static final int NUM_STATES = 16;
    private static final int TRANSITION_SPACE = NUM_STATES * NUM_STATES;

    // adj[digit][transition_id]
    private double[][] adj = new double[NUM_DIGITS][TRANSITION_SPACE];
    private long[] globalTransitionCount = new long[TRANSITION_SPACE];
    private long updatesCounter = 0;

    public RowMarkovNode(String id, CachedMarkovChainEvaluator evaluator,
            MultiSequenceExtractor extractor,
            Patch4x4FeedbackConfig feedbackConfig) {
        this.id = id;
        this.evaluator = evaluator;
        this.extractor = extractor;
        this.feedbackConfig = feedbackConfig;
        logger.debug("Created RowMarkovNode with id {}, feedback enabled={}", id, feedbackConfig.enabled);
    }

    public void setFeedbackConfig(Patch4x4FeedbackConfig config) {
        this.feedbackConfig = config;
    }

    public void resetFeedbackState() {
        adj = new double[NUM_DIGITS][TRANSITION_SPACE];
        globalTransitionCount = new long[TRANSITION_SPACE];
        updatesCounter = 0;
        logger.info("RowMarkovNode feedback state reset.");
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
        // Binarize and flatten
        int[][] binary2D = DigitMarkovModel.binarize(img.pixels, 128);
        byte[] binaryFlat = new byte[784];
        for (int r = 0; r < 28; r++) {
            for (int c = 0; c < 28; c++) {
                binaryFlat[r * 28 + c] = (byte) binary2D[r][c];
            }
        }

        double[] avgLogL = evaluator.evaluate(img.imageRelPath, img.imageHash, binaryFlat);

        // Apply feedback adjustment if enabled
        if (feedbackConfig.enabled) {
            int[] tids = extractTransitionIds(img);
            if (tids.length > 0) {
                // Calculate adjustment per digit
                double[] adjustments = new double[NUM_DIGITS];
                for (int d = 0; d < NUM_DIGITS; d++) {
                    double sumAdj = 0;
                    for (int tid : tids) {
                        sumAdj += adj[d][tid];
                    }
                    // Average adjustment matching scoring
                    adjustments[d] = sumAdj / tids.length;
                }

                // Add weighted adjustment
                for (int d = 0; d < NUM_DIGITS; d++) {
                    avgLogL[d] += (adjustments[d] * feedbackConfig.adjScale);
                }
            }
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
            logger.debug("RowMarkovNode {} [Label {}]: AvgLogL range=[{:.4f}, {:.4f}]",
                    id, img.label, minAvg, maxAvg);
        }

        return new NodeResult(avgLogL);
    }

    public void applyDecayIfEnabled(boolean isLearningAllowed) {
        if (feedbackConfig.enabled && feedbackConfig.learningEnabled && isLearningAllowed
                && feedbackConfig.applyDecayEachEpoch) {
            applyDecay();
            logger.info("Applied decay to RowMarkovNode adjustments (epoch end)");
        }
    }

    public Patch4x4FeedbackConfig getFeedbackConfig() {
        return feedbackConfig;
    }

    public int[] extractTransitionIds(DigitImage img) {
        // Re-extract using same logic as classifier/evaluator
        int[][] binary2D = DigitMarkovModel.binarize(img.pixels, 128);
        // The extractor expects 2D binary int array

        List<int[]> sequences = extractor.extractSequences(binary2D);

        // Count total transitions first
        int totalTransitions = 0;
        for (int[] seq : sequences) {
            if (seq.length > 1) {
                totalTransitions += (seq.length - 1);
            }
        }

        int[] tids = new int[totalTransitions];
        int idx = 0;
        for (int[] seq : sequences) {
            for (int i = 0; i < seq.length - 1; i++) {
                int from = seq[i];
                int to = seq[i + 1];
                // Validate range just in case
                if (from >= 0 && from < NUM_STATES && to >= 0 && to < NUM_STATES) {
                    tids[idx++] = from * NUM_STATES + to;
                }
            }
        }
        // Resize if any skipped (shouldn't happen with correct logic)
        if (idx != totalTransitions) {
            int[] result = new int[idx];
            System.arraycopy(tids, 0, result, 0, idx);
            return result;
        }
        return tids;
    }

    public void applyFeedback(int[] transitionIds, int trueDigit, int rivalDigit, boolean wasCorrect, double margin) {
        applyFeedback(transitionIds, trueDigit, rivalDigit, wasCorrect, margin, 1.0);
    }

    public void applyFeedback(int[] transitionIds, int trueDigit, int rivalDigit, boolean wasCorrect, double margin,
            double updateScale) {
        if (!feedbackConfig.enabled || !feedbackConfig.learningEnabled) {
            return;
        }

        // If external scale is 0, skip
        if (updateScale <= 0.0) {
            return;
        }

        // Gating
        if (feedbackConfig.updateOnlyIfIncorrect && wasCorrect) {
            return; // No update
        }
        if (feedbackConfig.useMarginGating && margin >= feedbackConfig.marginTarget) {
            return; // Margin satisfied
        }

        // Base Scale
        double baseScale = 1.0;
        if (feedbackConfig.useMarginGating) {
            baseScale = Math.max(0.0, Math.min(1.0, feedbackConfig.marginTarget - margin));
        }

        baseScale *= updateScale;

        // Updates
        for (int tid : transitionIds) {
            globalTransitionCount[tid]++;
            updatesCounter++;

            // Frequency Scaling
            double scaleFreq = 1.0;
            long count = globalTransitionCount[tid];
            if (feedbackConfig.frequencyScalingEnabled) {
                if ("GLOBAL_LINEAR".equalsIgnoreCase(feedbackConfig.frequencyScalingMode)) {
                    scaleFreq = 1.0 / (1.0 + count);
                } else {
                    // Default GLOBAL_SQRT
                    scaleFreq = 1.0 / Math.sqrt(1.0 + count);
                }
            }

            // Clamp scaleFreq
            if (scaleFreq < feedbackConfig.minUpdateScale)
                scaleFreq = feedbackConfig.minUpdateScale;
            if (scaleFreq > feedbackConfig.maxUpdateScale)
                scaleFreq = feedbackConfig.maxUpdateScale;

            double effEta = feedbackConfig.eta * baseScale * scaleFreq;

            // Symmetric update
            adj[trueDigit][tid] += effEta;
            adj[rivalDigit][tid] -= effEta;

            // Clamp absolute value
            if (adj[trueDigit][tid] > feedbackConfig.maxAdjAbs)
                adj[trueDigit][tid] = feedbackConfig.maxAdjAbs;
            if (adj[trueDigit][tid] < -feedbackConfig.maxAdjAbs)
                adj[trueDigit][tid] = -feedbackConfig.maxAdjAbs;

            if (adj[rivalDigit][tid] > feedbackConfig.maxAdjAbs)
                adj[rivalDigit][tid] = feedbackConfig.maxAdjAbs;
            if (adj[rivalDigit][tid] < -feedbackConfig.maxAdjAbs)
                adj[rivalDigit][tid] = -feedbackConfig.maxAdjAbs;

            // Decay logic
            if (feedbackConfig.applyDecayEveryNUpdates > 0 &&
                    updatesCounter % feedbackConfig.applyDecayEveryNUpdates == 0) {
                applyDecay();
            }
        }
    }

    private void applyDecay() {
        double factor = 1.0 - feedbackConfig.decayRate;
        for (int d = 0; d < NUM_DIGITS; d++) {
            for (int t = 0; t < TRANSITION_SPACE; t++) {
                adj[d][t] *= factor;
            }
        }
    }
}
