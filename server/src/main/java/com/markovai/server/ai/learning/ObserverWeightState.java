package com.markovai.server.ai.learning;

import java.util.HashMap;
import java.util.Map;
import java.util.TreeMap;
import java.util.concurrent.ConcurrentHashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import com.markovai.server.ai.hierarchy.FactorGraphBuilder.ObserverWeightsConfig;

/**
 * Manages learned parameters (Thetas) for Competitive Normalized Observer
 * Weighting.
 * 
 * Logic:
 * Theta_i += alpha * (Advantage_i)
 * w_i = exp(Theta_i / T) / Sum(exp(Theta_j / T)) * K
 */
public class ObserverWeightState {
    private static final Logger logger = LoggerFactory.getLogger(ObserverWeightState.class);

    private final Map<String, Double> thetas = new ConcurrentHashMap<>();
    private final ObserverWeightsConfig config;

    // Config Caches
    private final double alpha;
    private final double temperature;
    private final boolean usePayoffScale;
    private final double marginClip;
    private final double l2;

    private int totalUpdates = 0;

    public ObserverWeightState(ObserverWeightsConfig config) {
        this.config = config;
        this.alpha = config.alpha != null ? config.alpha : 0.01;
        this.temperature = config.temperature != null ? config.temperature : 1.0;
        this.usePayoffScale = config.usePayoffScale != null ? config.usePayoffScale : true;
        this.marginClip = config.marginClip != null ? config.marginClip : 1.0;
        this.l2 = config.l2 != null ? config.l2 : 0.0;
    }

    public void reset() {
        thetas.clear();
        totalUpdates = 0;
    }

    /**
     * Updates the Theta parameter for a specific node based on its advantage.
     * Theta += alpha * scale * advantage
     */
    public void update(String nodeId, double rawMargin, double payoffScale) {
        double advantage = computeAdvantage(rawMargin);
        double scale = usePayoffScale ? payoffScale : 1.0;
        double delta = alpha * scale * advantage;

        thetas.compute(nodeId, (k, v) -> {
            double old = (v == null) ? 0.0 : v;
            return (old + delta) * (1.0 - l2);
        });
        totalUpdates++;
    }

    private double computeAdvantage(double margin) {
        // Clip margin
        double clipped = Math.max(-marginClip, Math.min(marginClip, margin));

        if ("scaled_margin".equalsIgnoreCase(config.advantageMode)) {
            return clipped / marginClip;
        } else {
            // Default "signed_margin"
            return Math.signum(clipped);
        }
    }

    /**
     * Updates Theta using Cross-Entropy Gradient Ascent.
     * Advantage = s_i[true] - E_p[s_i]
     * Theta += alpha * scale * Advantage
     *
     * @param nodeId         Node ID
     * @param centeredScores s_i centered by image mean
     * @param trueDigit      The ground truth label
     * @param probs          Ensemble probabilities p[d]
     * @param payoffScale    Additional scale factor (e.g. from payoff)
     */
    public void updateCrossEntropy(String nodeId, double[] centeredScores, int trueDigit, double[] probs,
            double payoffScale) {
        // 1. Calculate Expected Value of observer score under ensemble distribution
        double expected = 0.0;
        for (int d = 0; d < 10; d++) {
            expected += probs[d] * centeredScores[d];
        }

        // 2. Advantage: Score of True Digit - Expected Score
        double advantage = centeredScores[trueDigit] - expected;

        // 3. Update
        double scale = usePayoffScale ? payoffScale : 1.0;
        // Note: For CE, gradients can be large. alpha is usually smaller.
        double delta = alpha * scale * advantage;

        thetas.compute(nodeId, (k, v) -> {
            double old = (v == null) ? 0.0 : v;
            return (old + delta) * (1.0 - l2);
        });
        totalUpdates++;
    }

    /**
     * Computes normalized weights from current Thetas using Softmax.
     * w_i = (exp(theta_i/T) / Sum) * K
     * 
     * @param activeNodeIds The set of nodes to consider for normalization.
     * @return Map of NodeId -> Weight
     */
    public Map<String, Double> computeWeights(java.util.Set<String> activeNodeIds) {
        Map<String, Double> weights = new HashMap<>();
        if (activeNodeIds == null || activeNodeIds.isEmpty())
            return weights;

        // 1. Compute Exponentials and Sum
        double sumExp = 0.0;
        Map<String, Double> exps = new HashMap<>();

        for (String id : activeNodeIds) {
            double theta = thetas.getOrDefault(id, 0.0);
            double val = Math.exp(theta / temperature);
            exps.put(id, val);
            sumExp += val;
        }

        // 2. Determine Scale K
        double K = activeNodeIds.size(); // Default: NUM_OBSERVERS
        if (config.scaleK != null && !config.scaleK.equalsIgnoreCase("NUM_OBSERVERS")) {
            try {
                K = Double.parseDouble(config.scaleK);
            } catch (NumberFormatException e) {
                // Fallback to num observers
            }
        }

        // 3. Normalize
        if (sumExp > 0) {
            for (Map.Entry<String, Double> e : exps.entrySet()) {
                weights.put(e.getKey(), (e.getValue() / sumExp) * K);
            }
        } else {
            // Fallback (shouldn't happen with exp)
            for (String id : activeNodeIds) {
                weights.put(id, 1.0);
            }
        }

        return weights;
    }

    public void logSummary(String prefix, boolean fullDetails) {
        if (thetas.isEmpty()) {
            logger.info("{}: No params learned yet.", prefix);
            return;
        }

        double minT = Double.MAX_VALUE;
        double maxT = -Double.MAX_VALUE;
        double sumT = 0.0;

        for (double t : thetas.values()) {
            if (t < minT)
                minT = t;
            if (t > maxT)
                maxT = t;
            sumT += t;
        }
        double meanT = sumT / thetas.size();

        // Compute representative weights (assuming all present are active)
        Map<String, Double> currentWeights = computeWeights(thetas.keySet());

        logger.info("{}: Count={} Updates={} Theta[Mean={:.3f} Min={:.3f} Max={:.3f}]",
                prefix, thetas.size(), totalUpdates, meanT, minT, maxT);

        if (fullDetails) {
            // Sort by weight
            TreeMap<Double, String> sorted = new TreeMap<>();
            for (Map.Entry<String, Double> e : currentWeights.entrySet()) {
                sorted.put(e.getValue(), e.getKey()); // Value maps to Key (collision possible but rare enough for log)
                                                      // (Actually safer to list)
            }

            // Better sort safely
            StringBuilder sb = new StringBuilder();
            currentWeights.entrySet().stream()
                    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                    .forEach(e -> sb.append(String.format("%s=%.3f(th:%.2f) ", e.getKey(), e.getValue(),
                            thetas.getOrDefault(e.getKey(), 0.0))));

            logger.info("{} Weights: [{}]", prefix, sb.toString().trim());
        }
    }
}
