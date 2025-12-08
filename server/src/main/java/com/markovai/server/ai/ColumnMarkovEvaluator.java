package com.markovai.server.ai;

import java.util.List;

public class ColumnMarkovEvaluator implements MarkovChainEvaluator {

    private final DigitMarkovModel model;
    private final MultiSequenceExtractor extractor;
    private final String version;

    public ColumnMarkovEvaluator(DigitMarkovModel model, MultiSequenceExtractor extractor, String version) {
        this.model = model;
        this.extractor = extractor;
        this.version = version;
    }

    @Override
    public String getChainType() {
        return "col";
    }

    @Override
    public String getChainVersion() {
        return version;
    }

    @Override
    public double[] computeScores(byte[] binary28x28) {
        int[][] binary = new int[28][28];
        for (int i = 0; i < 784; i++) {
            binary[i / 28][i % 28] = binary28x28[i];
        }

        List<int[]> sequences = extractor.extractSequences(binary);

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

        double[] avgLogL = new double[10];
        for (int d = 0; d < 10; d++) {
            avgLogL[d] = sumLogL[d] / totalSteps;
        }

        return avgLogL;
    }
}
