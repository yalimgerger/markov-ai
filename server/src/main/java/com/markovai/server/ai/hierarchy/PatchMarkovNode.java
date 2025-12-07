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

        double[] ll = new double[10];
        for (int d = 0; d < 10; d++) {
            ll[d] = model.logLikelihood(d, sequence);
        }

        logger.debug("PatchMarkovNode {} computed log-likelihoods for image label {}", id, img.label);

        return new NodeResult(ll);
    }
}
