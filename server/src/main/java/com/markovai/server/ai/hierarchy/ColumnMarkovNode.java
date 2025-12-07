package com.markovai.server.ai.hierarchy;

import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.DigitMarkovModel;
import com.markovai.server.ai.MultiSequenceExtractor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class ColumnMarkovNode implements DigitFactorNode {

    private static final Logger logger = LoggerFactory.getLogger(ColumnMarkovNode.class);
    private final String id;
    private final DigitMarkovModel model;
    private final MultiSequenceExtractor extractor;

    public ColumnMarkovNode(String id, DigitMarkovModel model, MultiSequenceExtractor extractor) {
        this.id = id;
        this.model = model;
        this.extractor = extractor;
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
        int[][] binary = DigitMarkovModel.binarize(img.pixels, 128);
        List<int[]> sequences = extractor.extractSequences(binary);

        // 1. Compute Raw Log-Likelihoods & Total Steps
        double[] sumLogL = new double[10];
        long totalSteps = 0;
        for (int[] seq : sequences) {
            if (seq.length > 1) {
                totalSteps += (seq.length - 1);
            }
        }

        if (totalSteps == 0)
            totalSteps = 1;

        for (int d = 0; d < 10; d++) {
            sumLogL[d] = model.logLikelihoodForSequences(d, sequences);
        }

        // 2. Average per-step Log-Likelihood
        double[] avgLogL = new double[10];
        for (int d = 0; d < 10; d++) {
            avgLogL[d] = sumLogL[d] / totalSteps;
        }

        // 3. Debug Logging
        if (logger.isDebugEnabled()) {
            double minAvg = Double.MAX_VALUE;
            double maxAvg = Double.MIN_VALUE;
            for (double val : avgLogL) {
                if (val < minAvg)
                    minAvg = val;
                if (val > maxAvg)
                    maxAvg = val;
            }
            logger.debug("ColumnMarkovNode {} [Label {}]: Steps={}. AvgLogL range=[{:.4f}, {:.4f}]",
                    id, img.label, totalSteps, minAvg, maxAvg);
        }

        return new NodeResult(avgLogL);
    }
}
