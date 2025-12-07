package com.markovai.server.ai.hierarchy;

public class NodeResult {
    // length should be exactly 10, one entry per digit 0..9
    public final double[] logLikelihoodsPerDigit;

    public NodeResult(double[] logLikelihoodsPerDigit) {
        if (logLikelihoodsPerDigit == null || logLikelihoodsPerDigit.length != 10) {
            throw new IllegalArgumentException("logLikelihoodsPerDigit must be length 10");
        }
        this.logLikelihoodsPerDigit = logLikelihoodsPerDigit;
    }
}
