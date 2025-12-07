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

    public WeightedSumNode(String id, List<DigitFactorNode> children, Map<String, Double> weights) {
        this.id = id;
        if (children != null) {
            this.children.addAll(children);
        }
        this.weights = weights;
    }

    public void addChild(DigitFactorNode child) {
        this.children.add(child);
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
            if (cr == null) {
                // Should not happen if traversed correctly
                logger.warn("Missing result for child {}", child.getId());
                continue;
            }

            double w = weights.getOrDefault(child.getId(), 1.0);
            for (int d = 0; d < 10; d++) {
                totalLogL[d] += w * cr.logLikelihoodsPerDigit[d];
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
            logger.trace("WeightedSumNode {} combined {} children. Best Digit: {} ({:.2f})", id, children.size(), bestD,
                    bestL);
        }

        return new NodeResult(totalLogL);
    }
}
