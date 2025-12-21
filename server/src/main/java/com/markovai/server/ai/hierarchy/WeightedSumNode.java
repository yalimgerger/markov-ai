package com.markovai.server.ai.hierarchy;

import com.markovai.server.ai.DigitImage;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

public class WeightedSumNode implements DigitFactorNode {
    private static final Logger logger = LoggerFactory.getLogger(WeightedSumNode.class);

    private final String id;
    private final List<DigitFactorNode> children = new ArrayList<>();
    private final Map<String, Double> weights;
    private Map<String, Double> weightOverride = null;
    private boolean standardizeObserverScores = false;

    public WeightedSumNode(String id, List<DigitFactorNode> children, Map<String, Double> weights) {
        this.id = id;
        if (children != null) {
            this.children.addAll(children);
        }
        this.weights = weights;
    }

    public void setStandardizeObserverScores(boolean s) {
        this.standardizeObserverScores = s;
        logger.info("WeightedSumNode: standardizeObserverScores={}", s);
    }

    public boolean isStandardizeObserverScores() {
        return standardizeObserverScores;
    }

    public void setWeightOverride(Map<String, Double> override) {
        this.weightOverride = override;
    }

    public void addChild(DigitFactorNode child) {
        this.children.add(child);
    }

    public double getWeight(String childId) {
        if (weightOverride != null && weightOverride.containsKey(childId)) {
            return weightOverride.get(childId);
        }
        return weights.getOrDefault(childId, 0.0);
    }

    @Override
    public String getId() {
        return id;
    }

    @Override
    public List<DigitFactorNode> getChildren() {
        return children;
    }

    @Override
    public NodeResult computeForImage(DigitImage img, Map<String, NodeResult> childResults) {
        double[] totalLogL = new double[10];

        for (DigitFactorNode child : children) {
            NodeResult cr = childResults.get(child.getId());
            double w = getWeight(child.getId());

            if (cr == null) {
                if (w == 0.0) {
                    continue;
                }
                logger.warn("Missing result for child {}", child.getId());
                continue;
            }

            double[] scores = cr.logLikelihoodsPerDigit;
            if (standardizeObserverScores) {
                double mean = 0.0;
                for (double v : scores)
                    mean += v;
                mean /= 10.0;

                double var = 0.0;
                for (double v : scores)
                    var += (v - mean) * (v - mean);
                double std = Math.sqrt(var / 10.0) + 1e-6; // Add eps

                // Use standardized scores for fusion
                double[] z = new double[10];
                for (int d = 0; d < 10; d++) {
                    z[d] = (scores[d] - mean) / std;
                }
                scores = z;
            }

            for (int d = 0; d < 10; d++) {
                totalLogL[d] += w * scores[d];
            }
        }

        if (logger.isTraceEnabled()) {
            // Find best for logging
            int bestD = -1;
            double bestL = Double.NEGATIVE_INFINITY;
            for (int d = 0; d < 10; d++) {
                if (totalLogL[d] > bestL) {
                    bestL = totalLogL[d];
                    bestD = d;
                }
            }
            logger.trace("WeightedSumNode {} combined {} children. Best Digit: {} ({})", id, children.size(), bestD,
                    String.format("%.2f", bestL));
        }

        return new NodeResult(totalLogL);
    }
}
