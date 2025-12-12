package com.markovai.server.ai;

import java.util.ArrayList;
import java.util.List;

public class Patch4x4Evaluator implements MarkovChainEvaluator {

    private final DigitPatch4x4UnigramModel model;
    private final double smoothingLambda;
    private final String version;

    public Patch4x4Evaluator(DigitPatch4x4UnigramModel model, double smoothingLambda, String version) {
        this.model = model;
        this.smoothingLambda = smoothingLambda;
        this.version = version;
    }

    @Override
    public String getChainType() {
        return "patch4x4";
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

        List<Integer> patchSymbols = new ArrayList<>();
        for (int r = 0; r < 7; r++) {
            for (int c = 0; c < 7; c++) {
                patchSymbols.add(DigitPatch4x4UnigramModel.encodePatch(binary, r * 4, c * 4));
            }
        }
        int nSteps = patchSymbols.size(); // Should be 49

        double[] smoothedSum = new double[10];

        for (int d = 0; d < 10; d++) {
            double prevLp = 0.0;
            boolean first = true;
            double sum = 0.0;

            for (Integer symbol : patchSymbols) {
                double lp = model.logProbForSymbol(d, symbol);

                if (first) {
                    sum += lp;
                    first = false;
                } else {
                    double diff = lp - prevLp;
                    sum += lp - smoothingLambda * Math.abs(diff);
                }
                prevLp = lp;
            }
            smoothedSum[d] = sum;
        }

        double[] avgLogL = new double[10];
        for (int d = 0; d < 10; d++) {
            avgLogL[d] = smoothedSum[d] / nSteps;
        }

        return avgLogL;
    }
}
