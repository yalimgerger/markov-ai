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
        return new ClassificationResult(bestDigit, totalLogL, null);
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

    public double evaluateAccuracy(List<DigitImage> testData, boolean isTestSet) {
        // Default behavior: use internal classification (equivalent to Layered)
        // We do strictly internal logic here to avoid dependency cycles or overhead
        return evaluateAccuracyInternal(testData, isTestSet, null);
    }

    public double evaluateAccuracy(List<DigitImage> testData, boolean isTestSet,
            com.markovai.server.ai.inference.InferenceEngine engine) {
        return evaluateAccuracyInternal(testData, isTestSet, engine);
    }

    private double evaluateAccuracyInternal(List<DigitImage> testData, boolean isTestSet,
            com.markovai.server.ai.inference.InferenceEngine engine) {
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

        // Try to apply decay at start of epoch/eval if applicable
        // Only allow decay if this is NOT a test set (i.e. isLearningAllowed =
        // !isTestSet)
        if (p4Node != null) {
            p4Node.applyDecayIfEnabled(!isTestSet);
        }
        if (rowNode != null) {
            rowNode.applyDecayIfEnabled(!isTestSet);
        }
        if (colNode != null) {
            colNode.applyDecayIfEnabled(!isTestSet);
        }

        int correct = 0;
        int total = 0;
        int p4Updates = 0;
        int rowUpdates = 0;
        int colUpdates = 0;

        for (DigitImage img : testData) {
            int predicted;
            double[] scores;

            if (engine != null) {
                com.markovai.server.ai.inference.InferenceResult result = engine.infer(img);
                predicted = result.getPredictedDigit();
                scores = result.getScores();
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

            // Feedback Loop
            if (p4Node != null) {
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
                boolean wasCorrect = (predicted == trueDigit);

                // Patch4x4 Feedback
                int[] symbols = p4Node.extractPatchSymbols(img);
                p4Node.applyFeedback(symbols, trueDigit, rivalDigit, wasCorrect, margin);
                if (p4Node.getFeedbackConfig().learningEnabled) {
                    p4Updates++;
                }

                // Row Feedback
                if (rowNode != null) {
                    int[] tids = rowNode.extractTransitionIds(img);
                    rowNode.applyFeedback(tids, trueDigit, rivalDigit, wasCorrect, margin);
                    if (rowNode.getFeedbackConfig().learningEnabled) {
                        rowUpdates++;
                    }
                }

                // Column Feedback
                if (colNode != null) {
                    int[] tids = colNode.extractTransitionIds(img);
                    colNode.applyFeedback(tids, trueDigit, rivalDigit, wasCorrect, margin);
                    if (colNode.getFeedbackConfig().learningEnabled) {
                        colUpdates++;
                    }
                }
            }

            if (total % 1000 == 0) {
                logger.info("MRF Evaluated {}/{}...", total, testData.size());
            }
        }

        double accuracy = (double) correct / total;
        if (p4Node != null && p4Node.getFeedbackConfig().enabled) {
            logger.info("MRF Evaluation complete. Accuracy: {} ({}/{}) [P4Updates: {}, RowUpdates: {}, ColUpdates: {}]",
                    String.format("%.4f", accuracy), correct, total, p4Updates, rowUpdates, colUpdates);
        } else {
            logger.info("MRF Evaluation complete. Accuracy: {} ({}/{})",
                    String.format("%.4f", accuracy), correct, total);
        }
        return accuracy;
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
