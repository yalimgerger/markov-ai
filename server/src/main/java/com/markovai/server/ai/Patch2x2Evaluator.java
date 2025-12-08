package com.markovai.server.ai;

public class Patch2x2Evaluator implements MarkovChainEvaluator {

    private final DigitMarkovModel model;
    private final SequenceExtractor extractor;
    private final String version;

    public Patch2x2Evaluator(DigitMarkovModel model, SequenceExtractor extractor, String version) {
        this.model = model;
        this.extractor = extractor;
        this.version = version;
    }

    @Override
    public String getChainType() {
        return "patch2x2";
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

        int[] sequence = extractor.extractSequence(binary);

        double[] sumLogL = new double[10];
        long totalSteps = (sequence.length > 1) ? (sequence.length - 1) : 1;

        for (int d = 0; d < 10; d++) {
            sumLogL[d] = model.logLikelihood(d, sequence);
        }

        double[] avgLogL = new double[10];
        for (int d = 0; d < 10; d++) {
            avgLogL[d] = sumLogL[d] / totalSteps;
        }

        return avgLogL;
    }
}
