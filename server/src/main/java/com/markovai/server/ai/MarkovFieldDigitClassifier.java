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
        logger.info("Evaluating MRF accuracy on {} images (isTestSet={})...", testData.size(), isTestSet);

        com.markovai.server.ai.hierarchy.Patch4x4Node p4Node = findPatch4x4Node();

        // LEAKAGE GUARD
        if (isTestSet && p4Node != null && p4Node.getFeedbackConfig().learningEnabled) {
            throw new IllegalStateException(
                    "LEAKAGE PREVENTION: Feedback learning is active but dataset is marked as TEST. Stopping evaluation to prevent contamination.");
        }

        // Try to apply decay at start of epoch/eval if applicable
        // Only allow decay if this is NOT a test set (i.e. isLearningAllowed =
        // !isTestSet)
        if (p4Node != null) {
            p4Node.applyDecayIfEnabled(!isTestSet);
        }

        int correct = 0;
        int total = 0;
        int feedbackUpdates = 0;

        for (DigitImage img : testData) {
            ClassificationResult result = classifyWithDetails(img);
            int predicted = result.getPredictedDigit();
            int trueDigit = img.label;

            if (predicted == trueDigit) {
                correct++;
            }
            total++;

            // Feedback Loop
            if (p4Node != null) {
                double[] scores = result.getLogLikelihoods();
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

                int[] symbols = p4Node.extractPatchSymbols(img);
                p4Node.applyFeedback(symbols, trueDigit, rivalDigit, wasCorrect, margin);
                if (p4Node.getFeedbackConfig().learningEnabled) {
                    feedbackUpdates++;
                }
            }

            if (total % 1000 == 0) {
                logger.info("MRF Evaluated {}/{}...", total, testData.size());
            }
        }

        double accuracy = (double) correct / total;
        if (p4Node != null && p4Node.getFeedbackConfig().enabled) {
            logger.info("MRF Evaluation complete. Accuracy: {} ({}/{}) [Feedback Updates: {}]",
                    String.format("%.4f", accuracy), correct, total, feedbackUpdates);
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

    private com.markovai.server.ai.hierarchy.Patch4x4Node findPatch4x4Node() {
        return findNodeRecursive(root, "patch4x4");
    }

    private com.markovai.server.ai.hierarchy.Patch4x4Node findNodeRecursive(DigitFactorNode current, String targetId) {
        if (current instanceof com.markovai.server.ai.hierarchy.Patch4x4Node && current.getId().equals(targetId)) {
            return (com.markovai.server.ai.hierarchy.Patch4x4Node) current;
        }
        for (DigitFactorNode child : current.getChildren()) {
            com.markovai.server.ai.hierarchy.Patch4x4Node found = findNodeRecursive(child, targetId);
            if (found != null)
                return found;
        }
        return null;
    }
}
