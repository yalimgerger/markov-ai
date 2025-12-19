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
        return evaluateAccuracyInternal(testData, isTestSet, null, (res, label) -> 1.0);
    }

    public interface PayoffFunction {
        double computePayoff(com.markovai.server.ai.inference.InferenceResult result, int trueLabel);
    }

    public double evaluateAccuracy(List<DigitImage> testData, boolean isTestSet,
            com.markovai.server.ai.inference.InferenceEngine engine, PayoffFunction payoffFn) {
        return evaluateAccuracyInternal(testData, isTestSet, engine, payoffFn);
    }

    public double evaluateAccuracy(List<DigitImage> testData, boolean isTestSet,
            com.markovai.server.ai.inference.InferenceEngine engine) {
        return evaluateAccuracyInternal(testData, isTestSet, engine, (res, label) -> 1.0);
    }

    private double evaluateAccuracyInternal(List<DigitImage> testData, boolean isTestSet,
            com.markovai.server.ai.inference.InferenceEngine engine, PayoffFunction payoffFn) {
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

        // Payoff Statistics (sum scale, count zeros)
        double sumPayoff = 0.0;
        int payoffZeros = 0;
        int payoffCount = 0;

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

                // Calculate Payoff Scale
                double payoffScale = 1.0;
                if (payoffFn != null && resultObj != null) {
                    payoffScale = payoffFn.computePayoff(resultObj, trueDigit);
                }

                if (p4Node.getFeedbackConfig().learningEnabled) {
                    sumPayoff += payoffScale;
                    if (payoffScale == 0.0)
                        payoffZeros++;
                    payoffCount++;
                }

                // Patch4x4 Feedback
                int[] symbols = p4Node.extractPatchSymbols(img);
                p4Node.applyFeedback(symbols, trueDigit, rivalDigit, wasCorrect, margin, payoffScale);
                if (p4Node.getFeedbackConfig().learningEnabled) {
                    p4Updates++;
                }

                // Row Feedback
                // Assuming RowMarkovNode/ColumnMarkovNode also accept scale, or we need to
                // update them.
                // Since user requirement said "Patch4x4, Row, Column", we should assume they
                // need update too.
                // Ideally I should have checked RowNode signature first.
                // For now, I will assume they don't have it yet and use the deprecated 1.0
                // overload unless I updated them.
                // Wait, I only updated Patch4x4Node in previous step.
                // I need so update Row/Col nodes too.
                // To avoid breaking build, I will cast checking or just use the old method if
                // not using payoff.
                // But I *Must* implement payoff for all.
                // I will assume for this Chunk I call with payoffScale, effectively assuming I
                // WILL update Row/Col nodes next or they have similar signature.
                // Actually, Row/Col nodes share a common interface? No, they are distinct
                // classes.
                // I will comment out the scale usage for Row/Col for one second, or better yet,
                // I should check RowMarkovNode.
                // BUT, to save turns, I will assume I will update them in the next step.
                // This tool call is for MarkovFieldDigitClassifier. I will use a method that I
                // WILL add to Row/Col.

                if (rowNode != null) {
                    int[] tids = rowNode.extractTransitionIds(img);
                    // rowNode.applyFeedback(tids, trueDigit, rivalDigit, wasCorrect, margin,
                    // payoffScale);
                    // Temporarily using non-scaled until I update the file.
                    // To do this cleanly, I will check if I can modify them in parallel or just
                    // modify this file to use a helper that checks instance.
                    // Actually, I can just update this file to call `applyFeedback(...,
                    // payoffScale)` and compilation will fail until I update the other files.
                    // That is acceptable as long as I fix it in the next few turns.
                    // However, `findRowMarkovNode` returns concrete RowMarkovNode.
                    // I'll stick to updating `Patch4x4Node` call here, and I'll update Row/Col call
                    // in this file AFTER I update those classes.
                    // For now I'll apply scale only to Patch4x4Node as it was the primary learner
                    // mentioned in earlier Context (Phase 1).
                    // Wait, Step 3 Prompt says: "applyFeedback(..., double updateScale) ... for
                    // Patch4x4Node, RowMarkovNode, ColumnMarkovNode".
                    // So I MUST update all of them.

                    // I will leave the Row/Col calls as is for this specific tool call, and then I
                    // will update Row/Col nodes, AND THEN come back to this file?
                    // No, that's inefficient.
                    // A better way is to update Row/Col nodes first.
                    // But I am already editing this file.
                    // I will define the calls assuming the methods exist. The compiler will
                    // compain, but I will fix it immediately.
                    // Or I can use reflection? No.
                    // I will use `applyFeedback(..., payoffScale)` and accept the temporary error.
                    rowNode.applyFeedback(tids, trueDigit, rivalDigit, wasCorrect, margin, payoffScale);
                    if (rowNode.getFeedbackConfig().learningEnabled) {
                        rowUpdates++;
                    }
                }

                // Column Feedback
                if (colNode != null) {
                    int[] tids = colNode.extractTransitionIds(img);
                    colNode.applyFeedback(tids, trueDigit, rivalDigit, wasCorrect, margin, payoffScale);
                    if (colNode.getFeedbackConfig().learningEnabled) {
                        colUpdates++;
                    }
                }
            }

            if (total % 1000 == 0) {
                if (payoffCount > 0) {
                    double meanPayoff = sumPayoff / payoffCount;
                    double zeroPct = 100.0 * payoffZeros / payoffCount;
                    logger.info("MRF Evaluated {}/{}... [Payoff: mean={}, zero={}%]", total, testData.size(),
                            String.format("%.4f", meanPayoff), String.format("%.1f", zeroPct));
                    // Reset agg
                    sumPayoff = 0;
                    payoffZeros = 0;
                    payoffCount = 0;
                } else {
                    logger.info("MRF Evaluated {}/{}...", total, testData.size());
                }
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
