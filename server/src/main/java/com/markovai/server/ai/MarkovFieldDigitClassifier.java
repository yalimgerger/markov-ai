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
    private final Map<String, DigitFactorNode> allNodes;

    public MarkovFieldDigitClassifier(DigitFactorNode root, Map<String, DigitFactorNode> allNodes) {
        this.root = root;
        this.allNodes = allNodes;
    }

    public int classify(DigitImage img) {
        // Compute bottom-up results
        // For a general DAG, topological sort is best.
        // For simplicity and our tree structure, recursive memoization works well.
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

        return bestDigit;
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

    public void evaluateAccuracy(List<DigitImage> testData) {
        logger.info("Evaluating MRF accuracy on {} test images...", testData.size());
        int correct = 0;
        int total = 0;

        for (DigitImage img : testData) {
            int predicted = classify(img);
            if (predicted == img.label) {
                correct++;
            }
            total++;
            if (total % 1000 == 0) {
                logger.info("MRF Evaluated {}/{}...", total, testData.size());
            }
        }

        double accuracy = (double) correct / total;
        logger.info("MRF Evaluation complete. Accuracy: {:.4f} ({}/{})", accuracy, correct, total);
    }
}
