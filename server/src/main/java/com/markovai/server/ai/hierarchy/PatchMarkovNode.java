package com.markovai.server.ai.hierarchy;

import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.DigitMarkovModel;
import com.markovai.server.ai.SequenceExtractor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class PatchMarkovNode implements DigitFactorNode {

    private static final Logger logger = LoggerFactory.getLogger(PatchMarkovNode.class);
    private final String id;
    private final DigitMarkovModel model;
    private final SequenceExtractor extractor;

    public PatchMarkovNode(String id, DigitMarkovModel model, SequenceExtractor extractor) {
        this.id = id;
        this.model = model;
        this.extractor = extractor;
        logger.debug("Created PatchMarkovNode with id {}", id);
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
        int[] sequence = extractor.extractSequence(binary);

        // 1. Compute Raw Log-Likelihoods & Total Steps
        double[] sumLogL = new double[10];
        long totalSteps = (sequence.length > 1) ? (sequence.length - 1) : 1;

        for (int d = 0; d < 10; d++) {
            sumLogL[d] = model.logLikelihood(d, sequence);
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
            logger.debug("PatchMarkovNode {} [Label {}]: Steps={}. AvgLogL range=[{:.4f}, {:.4f}]",
                    id, img.label, totalSteps, minAvg, maxAvg);
        }

        return new NodeResult(avgLogL);
    }
}
