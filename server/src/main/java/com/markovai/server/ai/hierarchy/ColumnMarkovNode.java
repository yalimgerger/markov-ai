package com.markovai.server.ai.hierarchy;

import com.markovai.server.ai.CachedMarkovChainEvaluator;
import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.DigitMarkovModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class ColumnMarkovNode implements DigitFactorNode {

    private static final Logger logger = LoggerFactory.getLogger(ColumnMarkovNode.class);
    private final String id;
    private final CachedMarkovChainEvaluator evaluator;

    public ColumnMarkovNode(String id, CachedMarkovChainEvaluator evaluator) {
        this.id = id;
        this.evaluator = evaluator;
        logger.debug("Created ColumnMarkovNode with id {}", id);
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

        if (logger.isDebugEnabled()) {
            double minAvg = Double.MAX_VALUE;
            double maxAvg = Double.MIN_VALUE;
            for (double val : avgLogL) {
                if (val < minAvg)
                    minAvg = val;
                if (val > maxAvg)
                    maxAvg = val;
            }
            logger.debug("ColumnMarkovNode {} [Label {}]: AvgLogL range=[{:.4f}, {:.4f}]",
                    id, img.label, minAvg, maxAvg);
        }

        return new NodeResult(avgLogL);
    }
}
