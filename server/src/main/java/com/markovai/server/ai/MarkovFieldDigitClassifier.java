package com.markovai.server.ai;

import com.markovai.server.ai.hierarchy.DigitFactorNode;
import com.markovai.server.ai.hierarchy.NodeResult;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

import com.markovai.server.ai.hierarchy.WeightedSumNode;

public class MarkovFieldDigitClassifier {
    private static final Logger logger = LoggerFactory.getLogger(MarkovFieldDigitClassifier.class);

    private final DigitFactorNode root;

    public MarkovFieldDigitClassifier(DigitFactorNode root) {
        this.root = root;
    }

    public java.util.Set<String> getObserverNodeIds() {
        java.util.Set<String> ids = new java.util.HashSet<>();
        if (root instanceof com.markovai.server.ai.hierarchy.WeightedSumNode) {
            for (com.markovai.server.ai.hierarchy.DigitFactorNode child : ((com.markovai.server.ai.hierarchy.WeightedSumNode) root)
                    .getChildren()) {
                ids.add(child.getId());
            }
        }
        return ids;
    }

    public ClassificationResult classifyWithDetails(DigitImage img) {
        // Compute bottom-up results
        Map<String, NodeResult> results = new HashMap<>();
        NodeResult rootRes = computeRecursive(root, img, results);

        double[] totalLogL = rootRes.logLikelihoodsPerDigit;
        int bestDigit = -1;
        double bestLogL = Double.NEGATIVE_INFINITY;

        for (int d = 0; d < 10; d++) {
            if (totalLogL[d] > bestLogL) {
                bestLogL = totalLogL[d];
                bestDigit = d;
            }
        }

        // Extract leaf scores
        Map<String, double[]> leafScores = new HashMap<>();
        // We want scores for all nodes, or just leaves?
        // Prompt says "for each leaf observer node".
        // In our graph, row, col, patch4x4 are children of root.
        // We can return all of them.
        // Only add observer nodes (children of root) to leafScores
        java.util.Set<String> obsIds = getObserverNodeIds();
        for (java.util.Map.Entry<String, com.markovai.server.ai.hierarchy.NodeResult> entry : results.entrySet()) {
            if (obsIds.contains(entry.getKey())) {
                leafScores.put(entry.getKey(), entry.getValue().logLikelihoodsPerDigit);
            }
        }

        return new ClassificationResult(bestDigit, totalLogL, null, leafScores);
    }

    public int classify(DigitImage img) {
        return classifyWithDetails(img).getPredictedDigit();
    }

    private NodeResult computeRecursive(DigitFactorNode node, DigitImage img, Map<String, NodeResult> memo) {
        if (memo.containsKey(node.getId())) {
            return memo.get(node.getId());
        }

        // Compute children first
        Map<String, NodeResult> childResults = new HashMap<>();
        for (DigitFactorNode child : node.getChildren()) {
            boolean skip = false;
            // Optimization: If parent is WeightedSumNode and weight is 0, skip calculation
            if (node instanceof WeightedSumNode) {
                WeightedSumNode wsn = (WeightedSumNode) node;
                if (wsn.getWeight(child.getId()) == 0.0) {
                    skip = true;
                }
            }

            if (!skip) {
                childResults.put(child.getId(), computeRecursive(child, img, memo));
            }
        }

        // Compute this node
        NodeResult res = node.computeForImage(img, childResults);
        memo.put(node.getId(), res);
        return res;
    }

    public void setObserverWeights(Map<String, Double> weights) {
        if (root instanceof WeightedSumNode) {
            ((WeightedSumNode) root).setWeightOverride(weights);
        } else {
            // If root isn't weighted sum (e.g. single node), we can't easily re-weight
            // children without a combinator.
            // But layered/network usually has WeightedSumNode root.
            // If not, we log warning or ignore.
            logger.warn("Root is not WeightedSumNode, cannot apply observer weight overrides.");
        }
    }

    public double evaluateAccuracy(List<DigitImage> testData, boolean isTestSet) {
        // Default behavior: use internal classification (equivalent to Layered)
        // We do strictly internal logic here to avoid dependency cycles or overhead
        return evaluateAccuracyInternal(testData, isTestSet, null, (img, res, classifier, trueDigit, scores) -> {
            // Default no-op or simple 1.0 logic?
            // Since PayoffFunction overload defaults to 1.0, and PayoffFunction wrapper
            // calls generic applier.
            // We want standard feedback with scale 1.0.
            // Using the strategy that mimics (res, label) -> 1.0:
            int rivalDigit = -1;
            double rivalScore = Double.NEGATIVE_INFINITY;
            for (int d = 0; d < 10; d++) {
                if (d == trueDigit)
                    continue;
                if (scores[d] > rivalScore) {
                    rivalScore = scores[d];
                    rivalDigit = d;
                }
            }
            double margin = scores[trueDigit] - rivalScore;
            boolean wasCorrect = (res != null) ? (res.getPredictedDigit() == trueDigit) : false;
            classifier.applyFeedbackToNodes(img, trueDigit, rivalDigit, 1.0, wasCorrect, margin);
        });
    }

    public interface PayoffFunction {
        double computePayoff(com.markovai.server.ai.inference.InferenceResult result, int trueLabel);
    }

    public interface FeedbackStrategy {
        void apply(DigitImage img, com.markovai.server.ai.inference.InferenceResult result,
                MarkovFieldDigitClassifier classifier, int trueDigit, double[] scores);
    }

    public void applyFeedbackToNodes(DigitImage img, int positiveTarget, int negativeTarget, double scale) {
        applyFeedbackToNodes(img, positiveTarget, negativeTarget, scale, true, 0.0);
    }

    public void applyFeedbackToNodes(DigitImage img, int positiveTarget, int negativeTarget, double scale,
            boolean wasCorrect, double margin) {
        com.markovai.server.ai.hierarchy.Patch4x4Node p4Node = findPatch4x4Node();
        com.markovai.server.ai.hierarchy.RowMarkovNode rowNode = findRowMarkovNode();
        com.markovai.server.ai.hierarchy.ColumnMarkovNode colNode = findColumnMarkovNode();

        // Patch4x4
        if (p4Node != null) {
            int[] symbols = p4Node.extractPatchSymbols(img);
            p4Node.applyFeedback(symbols, positiveTarget, negativeTarget, wasCorrect, margin, scale);
        }

        // Row
        if (rowNode != null) {
            int[] tids = rowNode.extractTransitionIds(img);
            rowNode.applyFeedback(tids, positiveTarget, negativeTarget, wasCorrect, margin, scale);
        }

        // Column
        if (colNode != null) {
            int[] tids = colNode.extractTransitionIds(img);
            colNode.applyFeedback(tids, positiveTarget, negativeTarget, wasCorrect, margin, scale);
        }
    }

    public double evaluateAccuracy(List<DigitImage> testData, boolean isTestSet,
            com.markovai.server.ai.inference.InferenceEngine engine, FeedbackStrategy strategy) {
        return evaluateAccuracyInternal(testData, isTestSet, engine, strategy);
    }

    private double evaluateAccuracyInternal(List<DigitImage> testData, boolean isTestSet,
            com.markovai.server.ai.inference.InferenceEngine engine, FeedbackStrategy strategy) {
        logger.info("Evaluating MRF accuracy on {} images (isTestSet={})...", testData.size(), isTestSet);

        com.markovai.server.ai.hierarchy.Patch4x4Node p4Node = findPatch4x4Node();
        com.markovai.server.ai.hierarchy.RowMarkovNode rowNode = findRowMarkovNode();
        com.markovai.server.ai.hierarchy.ColumnMarkovNode colNode = findColumnMarkovNode();

        // LEAKAGE GUARD
        if (isTestSet) {
            if (p4Node != null && p4Node.getFeedbackConfig().learningEnabled) {
                throw new IllegalStateException("LEAKAGE PREVENTION: Patch4x4 learning active on TEST set.");
            }
            if (rowNode != null && rowNode.getFeedbackConfig().learningEnabled) {
                throw new IllegalStateException("LEAKAGE PREVENTION: RowMarkov learning active on TEST set.");
            }
            if (colNode != null && colNode.getFeedbackConfig().learningEnabled) {
                throw new IllegalStateException("LEAKAGE PREVENTION: ColumnMarkov learning active on TEST set.");
            }
        }

        if (p4Node != null)
            p4Node.applyDecayIfEnabled(!isTestSet);
        if (rowNode != null)
            rowNode.applyDecayIfEnabled(!isTestSet);
        if (colNode != null)
            colNode.applyDecayIfEnabled(!isTestSet);

        int correct = 0;
        int total = 0;

        // Stats for default logging if needed
        int updates = 0;

        for (DigitImage img : testData) {
            int predicted;
            double[] scores;
            com.markovai.server.ai.inference.InferenceResult resultObj = null;

            if (engine != null) {
                resultObj = engine.infer(img);
                predicted = resultObj.getPredictedDigit();
                scores = resultObj.getScores();
            } else {
                ClassificationResult result = classifyWithDetails(img);
                predicted = result.getPredictedDigit();
                scores = result.getLogLikelihoods();
            }

            int trueDigit = img.label;
            if (predicted == trueDigit) {
                correct++;
            }
            total++;

            // Custom Feedback Strategy
            if (strategy != null) {
                strategy.apply(img, resultObj, this, trueDigit, scores);
                // We don't easily track updates per node here since strategy does it.
                // We'll trust strategy or logs.
            }

            // Progress Log
            if (total % 1000 == 0) {
                logger.info("MRF Evaluated {}/{}...", total, testData.size());
            }
        }

        double accuracy = (double) correct / total;
        logger.info("MRF Evaluation complete. Accuracy: {} ({}/{})", String.format("%.4f", accuracy), correct, total);
        return accuracy;
    }

    public double evaluateAccuracy(List<DigitImage> testData, boolean isTestSet,
            com.markovai.server.ai.inference.InferenceEngine engine, PayoffFunction payoffFn) {
        // Adapt old payoff function to new strategy
        return evaluateAccuracy(testData, isTestSet, engine, (img, res, classifier, trueDigit, scores) -> {
            // emulate old logic
            int rivalDigit = -1;
            double rivalScore = Double.NEGATIVE_INFINITY;
            for (int d = 0; d < 10; d++) {
                if (d == trueDigit)
                    continue;
                if (scores[d] > rivalScore) {
                    rivalScore = scores[d];
                    rivalDigit = d;
                }
            }
            double margin = scores[trueDigit] - rivalScore;
            boolean wasCorrect = (res != null) ? (res.getPredictedDigit() == trueDigit) : false; // Approx

            double payoffScale = 1.0;
            if (payoffFn != null && res != null) {
                payoffScale = payoffFn.computePayoff(res, trueDigit);
            }

            classifier.applyFeedbackToNodes(img, trueDigit, rivalDigit, payoffScale, wasCorrect, margin);
        });
    }

    public double evaluateAccuracy(List<DigitImage> testData, boolean isTestSet,
            com.markovai.server.ai.inference.InferenceEngine engine) {
        return evaluateAccuracy(testData, isTestSet, engine, (PayoffFunction) null); // Use adapter above
    }

    public void setPatch4x4Config(com.markovai.server.ai.Patch4x4FeedbackConfig config) {
        com.markovai.server.ai.hierarchy.Patch4x4Node node = findPatch4x4Node();
        if (node != null) {
            node.setFeedbackConfig(config);
            logger.info("Updated Patch4x4 feedback config: enabled={}, learningEnabled={}",
                    config.enabled, config.learningEnabled);
        } else {
            logger.warn("Cannot set Patch4x4 config: Node not found in graph.");
        }
    }

    public void setRowFeedbackConfig(com.markovai.server.ai.Patch4x4FeedbackConfig config) {
        com.markovai.server.ai.hierarchy.RowMarkovNode node = findRowMarkovNode();
        if (node != null) {
            node.setFeedbackConfig(config);
            logger.info("Updated RowMarkov feedback config: enabled={}, learningEnabled={}",
                    config.enabled, config.learningEnabled);
        } else {
            logger.warn("Cannot set RowMarkov config: Node not found in graph.");
        }
    }

    public void setColumnFeedbackConfig(com.markovai.server.ai.Patch4x4FeedbackConfig config) {
        com.markovai.server.ai.hierarchy.ColumnMarkovNode node = findColumnMarkovNode();
        if (node != null) {
            node.setFeedbackConfig(config);
            logger.info("Updated ColumnMarkov feedback config: enabled={}, learningEnabled={}",
                    config.enabled, config.learningEnabled);
        } else {
            logger.warn("Cannot set ColumnMarkov config: Node not found in graph.");
        }
    }

    private com.markovai.server.ai.hierarchy.Patch4x4Node findPatch4x4Node() {
        return (com.markovai.server.ai.hierarchy.Patch4x4Node) findNodeRecursive(root, "patch4x4",
                com.markovai.server.ai.hierarchy.Patch4x4Node.class);
    }

    private com.markovai.server.ai.hierarchy.RowMarkovNode findRowMarkovNode() {
        return (com.markovai.server.ai.hierarchy.RowMarkovNode) findNodeRecursive(root, "row",
                com.markovai.server.ai.hierarchy.RowMarkovNode.class);
    }

    private com.markovai.server.ai.hierarchy.ColumnMarkovNode findColumnMarkovNode() {
        return (com.markovai.server.ai.hierarchy.ColumnMarkovNode) findNodeRecursive(root, "col",
                com.markovai.server.ai.hierarchy.ColumnMarkovNode.class);
    }

    private <T> T findNodeRecursive(DigitFactorNode current, String targetId, Class<T> type) {
        if (current.getId().equals(targetId) && type.isInstance(current)) {
            return type.cast(current);
        }
        for (DigitFactorNode child : current.getChildren()) {
            T found = findNodeRecursive(child, targetId, type);
            if (found != null)
                return found;
        }
        return null;
    }
}
