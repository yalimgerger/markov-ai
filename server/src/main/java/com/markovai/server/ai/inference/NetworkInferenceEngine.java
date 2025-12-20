package com.markovai.server.ai.inference;

import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.MarkovFieldDigitClassifier;
import com.markovai.server.ai.ClassificationResult;
import com.markovai.server.ai.hierarchy.FactorGraphBuilder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class NetworkInferenceEngine implements InferenceEngine {

    private static final Logger logger = LoggerFactory.getLogger(NetworkInferenceEngine.class);

    private final MarkovFieldDigitClassifier classifier;
    private final FactorGraphBuilder.ConfigRoot fullConfig;
    private final FactorGraphBuilder.NetworkConfig config;

    public NetworkInferenceEngine(MarkovFieldDigitClassifier classifier, FactorGraphBuilder.ConfigRoot fullConfig) {
        this.classifier = classifier;
        this.fullConfig = fullConfig;
        this.config = (fullConfig != null) ? fullConfig.network : null;
    }

    @Override
    public InferenceResult infer(DigitImage img) {
        // 1. Base Classification (Feed-forward / Bottom-up)
        ClassificationResult base = classifier.classifyWithDetails(img);
        double[] sBase = base.getLogLikelihoods();

        if (config == null || config.enabled == null || !config.enabled) {
            // Fallback to layered behavior if network not enabled
            return new NetworkInferenceResult(
                    base.getPredictedDigit(),
                    sBase,
                    null,
                    1,
                    "network_disabled",
                    0.0, // initialEntropy (not computed)
                    0.0, // finalEntropy (not computed)
                    0.0, // finalMaxDelta (not computed)
                    false // oscillationDetected (not applicable)
            );
        }

        // 2. Initialize Belief
        double temp = config.temperature != null ? config.temperature : 1.0;
        double epsilon = config.epsilon != null ? config.epsilon : 1e-9;
        double stopEpsilon = config.stopEpsilon != null ? config.stopEpsilon : 1e-4;
        double priorWeight = config.priorWeight != null ? config.priorWeight : 0.25;
        double damping = config.damping != null ? config.damping : 0.5;
        int maxIters = config.maxIters != null ? config.maxIters : 8;

        double[] bPrev = MathUtil.softmax(sBase, temp);
        double[] sT = new double[10];
        double[] b = new double[10];

        int iters = 0;
        double finalMaxDelta = 0.0;
        double initialEntropy = MathUtil.entropy(bPrev);
        boolean oscillationDetected = false;
        double prevDelta = Double.MAX_VALUE;

        // 3. Attractor Loop
        java.util.List<double[]> trajectory = null;
        if (fullConfig != null && fullConfig.learning != null && fullConfig.learning.basin != null &&
                fullConfig.learning.basin.enabled != null && fullConfig.learning.basin.enabled &&
                fullConfig.learning.basin.useTrajectory != null && fullConfig.learning.basin.useTrajectory) {
            trajectory = new java.util.ArrayList<>();
            // Add initial belief
            trajectory.add(java.util.Arrays.copyOf(bPrev, 10)); // b0
        }

        for (int i = 0; i < maxIters; i++) {
            iters++;

            // S_t[d] = S_base[d] + priorWeight * log(b_prev[d] + epsilon)
            for (int d = 0; d < 10; d++) {
                sT[d] = sBase[d] + priorWeight * Math.log(bPrev[d] + epsilon);
            }

            // b_raw = softmax(S_t, temperature)
            double[] bRaw = MathUtil.softmax(sT, temp);

            // b = (1 - damping) * b_prev + damping * b_raw
            for (int d = 0; d < 10; d++) {
                b[d] = (1.0 - damping) * bPrev[d] + damping * bRaw[d];
            }

            // Capture trajectory step (b_t)
            if (trajectory != null) {
                trajectory.add(java.util.Arrays.copyOf(b, 10));
            }

            // Check convergence
            double delta = MathUtil.maxAbsDelta(b, bPrev);
            finalMaxDelta = delta;

            // Oscillation detection: delta increasing?
            // Simple heuristic: if delta increases significantly (e.g. > 1e-6 diff) it
            // might be oscillating/diverging.
            // Strict check: delta > prevDelta + 1e-9
            if (i > 0 && delta > prevDelta + 1e-9) {
                // If it increases twice in a row, we flag it.
                // But simplified: just flag if it *ever* increases significantly after first
                // iter.
                oscillationDetected = true;
            }
            prevDelta = delta;

            // Update bPrev
            System.arraycopy(b, 0, bPrev, 0, 10);

            if (delta < stopEpsilon) {
                break;
            }
        }

        // 4. Final Decision
        int predicted = MathUtil.argmax(b);
        double finalEntropy = MathUtil.entropy(b);

        // Debug Logging
        if (config.debugStats != null && config.debugStats) {
            // compute top1/top2 gap
            double max1 = -1.0;
            double max2 = -1.0;
            for (double v : b) {
                if (v > max1) {
                    max2 = max1;
                    max1 = v;
                } else if (v > max2) {
                    max2 = v;
                }
            }
            double gap = max1 - max2;

            logger.debug("NetworkInference: iters={} H_init={:.4f} H_final={:.4f} delta={:.6f} gap={:.4f} osc={}",
                    iters, initialEntropy, finalEntropy, finalMaxDelta, gap, oscillationDetected);
        }

        return new NetworkInferenceResult(
                predicted,
                sT, // The final refined scores (could also return sBase if we want raw evidence) -
                    // using sT reflects the prior influence
                b,
                iters,
                "network",
                initialEntropy,
                finalEntropy,
                finalMaxDelta,
                oscillationDetected,
                trajectory,
                base.getLeafScores());
    }
}
