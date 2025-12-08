package com.markovai.server.ai;

public interface MarkovChainEvaluator {
    String getChainType();

    String getChainVersion();

    double[] computeScores(byte[] binary28x28);
}
