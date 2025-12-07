package com.markovai.server.ai;

public class ClassificationResult {
    private final int predictedDigit;
    private final double[] logLikelihoods;
    private final double[] surprises;

    public ClassificationResult(int predictedDigit, double[] logLikelihoods, double[] surprises) {
        this.predictedDigit = predictedDigit;
        this.logLikelihoods = logLikelihoods;
        this.surprises = surprises;
    }

    public int getPredictedDigit() {
        return predictedDigit;
    }

    public double[] getLogLikelihoods() {
        return logLikelihoods;
    }

    public double[] getSurprises() {
        return surprises;
    }
}
