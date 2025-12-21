package com.markovai.server.ai.hierarchy;

import com.markovai.server.ai.DigitImage;
import com.markovai.server.ai.DigitGradientUnigramModel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Collections;
import java.util.List;
import java.util.Map;

public class GradientOrientationNode implements DigitFactorNode {
    private static final Logger logger = LoggerFactory.getLogger(GradientOrientationNode.class);

    private final String id;
    private final DigitGradientUnigramModel model;

    // We can allow debugging or detailed logging here
    private boolean debug = false;

    public GradientOrientationNode(String id, DigitGradientUnigramModel model) {
        this.id = id;
        this.model = model;
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
        // 1. Extract Symbols
        int[] symbols = model.extractSymbols(img.pixels);

        // 2. Score for each digit
        double[] logs = new double[10];

        for (int d = 0; d < 10; d++) {
            logs[d] = model.score(d, symbols);
        }

        // No feedback or learning in this step for this node (Unigram Base)

        return new NodeResult(logs);
    }

    public void setDebug(boolean d) {
        this.debug = d;
    }
}
