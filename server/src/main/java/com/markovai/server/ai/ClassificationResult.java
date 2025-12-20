package com.markovai.server.ai;

public class ClassificationResult {
    private final int predictedDigit;
    private final double[] logLikelihoods;
    private final double[] surprises;
    private final java.util.Map<String, double[]> leafScores;

    public ClassificationResult(int predictedDigit, double[] logLikelihoods, double[] surprises,
            java.util.Map<String, double[]> leafScores) {
        this.predictedDigit = predictedDigit;
        this.logLikelihoods = logLikelihoods;
        this.surprises = surprises;
        this.leafScores = leafScores;
    }

    // Legacy constructor for backward compatibility if needed, or update call sites
    public ClassificationResult(int predictedDigit, double[] logLikelihoods, double[] surprises) {
        this(predictedDigit, logLikelihoods, surprises, null);
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

    public java.util.Map<String, double[]> getLeafScores() {
        return leafScores;
    }
}
